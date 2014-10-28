package votepredictor;

import data.Vote;
import java.util.ArrayList;
import java.util.Collections;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;

/**
 *
 * @author vietan
 */
public class BayesianIdealPoint extends IdealPoint {

    protected double mean;
    protected double var;

    public BayesianIdealPoint() {
        this.name = "bayesian-ideal-point";
    }

    public BayesianIdealPoint(String name) {
        super(name);
    }

    @Override
    public void configure() {
        this.configure(5.0, 0.01, 1000, 0.0, 2.5);
    }

    @Override
    public void configure(double alpha, double eta, int maxIter) {
        this.configure(alpha, eta, maxIter, 0.0, 2.5);
    }

    public void configure(double alpha, double eta, int maxIter,
            double mu, double sigma) {
        super.configure(alpha, eta, maxIter);
        this.mean = mu;
        this.var = sigma;
    }

    @Override
    public String getName() {
        return this.name
                + "_a-" + MiscUtils.formatDouble(alpha)
                + "_e-" + MiscUtils.formatDouble(eta)
                + "_m-" + maxIter
                + "_m-" + MiscUtils.formatDouble(mean)
                + "_s-" + MiscUtils.formatDouble(var);
    }

    @Override
    protected void initialize() {
        ArrayList<RankingItem<Integer>> rankAuthors = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            int agreeCount = 0;
            int totalCount = 0;
            for (int bb = 0; bb < B; bb++) {
                if (mask[aa][bb]) {
                    if (votes[aa][bb] == Vote.WITH) {
                        agreeCount++;
                    }
                    totalCount++;
                }
            }
            double val = (double) agreeCount / totalCount;
            rankAuthors.add(new RankingItem<Integer>(aa, val));
        }
        Collections.sort(rankAuthors);
        posAnchor = rankAuthors.get(0).getObject();
        negAnchor = rankAuthors.get(rankAuthors.size() - 1).getObject();

        this.u = new double[A];
        for (int ii = 0; ii < A; ii++) {
            int aa = rankAuthors.get(ii).getObject();
            if (ii < A / 4) {
                this.u[aa] = SamplerUtils.getGaussian(anchorMean, anchorVar);
            } else if (ii > 3 * A / 4) {
                this.u[aa] = SamplerUtils.getGaussian(-anchorMean, anchorVar);
            } else {
                this.u[aa] = SamplerUtils.getGaussian(mean, var);
            }
        }

        this.x = new double[B];
        this.y = new double[B];
        for (int b = 0; b < B; b++) {
            this.x[b] = SamplerUtils.getGaussian(mean, var);
            this.y[b] = SamplerUtils.getGaussian(mean, var);
        }
    }

    @Override
    protected void iterate() {
        int stepSize = MiscUtils.getRoundStepSize(maxIter, 10);
        for (iter = 0; iter < maxIter; iter++) {
            if (verbose && iter % stepSize == 0) {
                double avgLlh = getLogLikelihood();
                logln("--- Iter " + iter + " / " + maxIter
                        + "\tllh = " + avgLlh
                        + ". +ive anchor (" + posAnchor + "): "
                        + MiscUtils.formatDouble(u[posAnchor])
                        + ". -ive anchor (" + negAnchor + "): "
                        + MiscUtils.formatDouble(u[negAnchor])
                        + ". A-rate: " + getLearningRate()
                        + ". B-rate: " + getLearningRate());
                if (Double.isNaN(avgLlh) || Double.isInfinite(avgLlh)) {
                    logln("Terminating ...");
                    return;
                }
            }
            updateUs();
            updateXYs();
        }
    }

    @Override
    protected void updateUs() {
        double aRate = getLearningRate();
        for (int a = 0; a < A; a++) {
            double grad = 0.0;
            // likelihood
            for (int b = 0; b < votes[a].length; b++) {
                if (mask[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    grad += x[b] * (votes[a][b] - prob); // only work for 0 and 1
                }
            }
            // prior
            grad -= (u[a] - mean) / var;
            // update
            u[a] += aRate * grad;
        }
    }

    @Override
    protected void updateXYs() {
        double bRate = getLearningRate();
        for (int b = 0; b < B; b++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int a = 0; a < A; a++) {
                if (mask[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    gradX += u[a] * (votes[a][b] - prob);
                    gradY += votes[a][b] - prob;
                }
            }
            // prior
            gradX -= (x[b] - mean) / var;
            gradY -= (y[b] - mean) / var;

            // update
            x[b] += bRate * gradX;
            y[b] += bRate * gradY;
        }
    }
}
