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
    private double rate;
    private ArrayList<Integer> authorList;
    private ArrayList<Integer> billList;
    private final double tolerance = 1E-7;

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

        double anchorVal = Math.sqrt(var);
        this.u = new double[A];
        for (int ii = 0; ii < A; ii++) {
            int aa = rankAuthors.get(ii).getObject();
            if (ii < A / 4) {
                this.u[aa] = SamplerUtils.getGaussian(anchorVal, anchorVar);
            } else if (ii > 3 * A / 4) {
                this.u[aa] = SamplerUtils.getGaussian(-anchorVal, anchorVar);
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
        this.authorList = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            this.authorList.add(aa);
        }
        this.billList = new ArrayList<>();
        for (int bb = 0; bb < B; bb++) {
            this.billList.add(bb);
        }
        
        double curval = 0.0;
        int stepSize = MiscUtils.getRoundStepSize(maxIter, 10);
        for (iter = 0; iter < maxIter; iter++) {
            rate = getAnnealingRate(eta, iter, maxIter);
            if (verbose && iter % stepSize == 0) {
                double absU = 0.0;
                double absX = 0.0;
                double absY = 0.0;
                for (int aa = 0; aa < A; aa++) {
                    absU += Math.abs(u[aa]);
                }
                for (int bb = 0; bb < B; bb++) {
                    absX += Math.abs(x[bb]);
                    absY += Math.abs(y[bb]);
                }

                double avgLlh = getLogLikelihood();
                logln("--- Iter " + iter + " / " + maxIter
                        + "\tllh = " + avgLlh
                        + ". +ive anchor (" + posAnchor + "): "
                        + MiscUtils.formatDouble(u[posAnchor])
                        + ". -ive anchor (" + negAnchor + "): "
                        + MiscUtils.formatDouble(u[negAnchor])
                        + ". rate: " + rate);
                double diff = Math.abs(curval - avgLlh);
                logln("--- --- absU: " + MiscUtils.formatDouble(absU / A)
                        + ". absX: " + MiscUtils.formatDouble(absX / B)
                        + ". absY: " + MiscUtils.formatDouble(absY / B)
                        + ". diff: " + diff);

                if (Double.isNaN(avgLlh) || Double.isInfinite(avgLlh)) {
                    logln("Terminating ...");
                    return;
                }

                if (diff < tolerance) {
                    logln("Diff = " + diff + ". Exiting int iteration " + iter
                            + ". Final llh = " + avgLlh);
                    break;
                } else {
                    curval = avgLlh;
                }

            }

            updateUs();
            updateXYs();
        }
    }

    protected double getAnnealingRate(double init, int t, int T) {
        return init / (1 + (double) t / T);
    }

    @Override
    protected void updateUs() {
        Collections.shuffle(authorList);
        for (int a : authorList) {
            if (!validAs[a]) {
                continue;
            }
            double llh = 0.0;
            for (int b = 0; b < votes[a].length; b++) {
                if (mask[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    llh += x[b] * (votes[a][b] - prob); // only work for 0 and 1
                }
            }
            u[a] += (llh - (u[a] - mean) / var) * rate / B;
        }
    }

    @Override
    protected void updateXYs() {
        Collections.shuffle(billList);
        for (int b : billList) {
            if (!validBs[b]) {
                continue;
            }
            double llhX = 0.0;
            double llhY = 0.0;
            for (int a = 0; a < A; a++) {
                if (mask[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    llhX += u[a] * (votes[a][b] - prob);
                    llhY += votes[a][b] - prob;
                }
            }
            x[b] += (llhX - (x[b] - mean) / var) * rate / A;
            y[b] += (llhY - (y[b] - mean) / var) * rate / A;
        }
    }
}
