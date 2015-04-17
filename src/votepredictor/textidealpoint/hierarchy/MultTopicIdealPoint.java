package votepredictor.textidealpoint.hierarchy;

import java.util.ArrayList;
import java.util.Collections;
import util.MiscUtils;
import util.SparseVector;
import votepredictor.BayesianIdealPoint;

/**
 *
 * @author vietan
 */
public class MultTopicIdealPoint extends BayesianIdealPoint {

    private final double tolerance = 1E-7;
    private int K;
    private SparseVector[] billThetas;

    private ArrayList<Integer> authorList;
    private ArrayList<Integer> billList;
    private ArrayList<Integer> topicList;
    private double rate;

    private double[][] us;

    public MultTopicIdealPoint() {
        this.name = "mult-topic-ideal-point";
    }

    public MultTopicIdealPoint(String name) {
        super(name);
    }

    public void configure(double alpha, double eta, int maxIter,
            double mu, double sigma, int K, SparseVector[] billThetas) {
        super.configure(alpha, eta, maxIter, mu, sigma);
        this.K = K;
        this.billThetas = billThetas;
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
        this.topicList = new ArrayList<>();
        for (int kk = 0; kk < K; kk++) {
            this.topicList.add(kk);
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
                        + ". A-rate: " + getLearningRate()
                        + ". B-rate: " + getLearningRate());
                logln("--- --- absU: " + MiscUtils.formatDouble(absU / A)
                        + ". absX: " + MiscUtils.formatDouble(absX / B)
                        + ". absY: " + MiscUtils.formatDouble(absY / B));
                if (Double.isNaN(avgLlh) || Double.isInfinite(avgLlh)) {
                    logln("Terminating ...");
                    return;
                }

                double diff = Math.abs(curval - avgLlh);
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

    @Override
    protected void updateUs() {
        Collections.shuffle(authorList);
        for (int aa : authorList) {
            if (!validAs[aa]) {
                continue;
            }
            Collections.shuffle(topicList);
            for (int kk : topicList) { // update 
                int count = 0;
                double llh = 0.0;
                for (int bb = 0; bb < B; bb++) {
                    if (mask[aa][bb]) {
                        double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                        double score = Math.exp(dotprod);
                        double prob = score / (1 + score);
                        llh += x[bb] * billThetas[bb].get(kk) * (votes[aa][bb] - prob);
                        count++;
                    }
                }
                us[aa][kk] += (llh / count - us[aa][kk] / var) * rate;
            }
        }
    }

    @Override
    public void updateXYs() {
        Collections.shuffle(billList);
        for (int bb : billList) {
            if (!validBs[bb]) {
                continue;
            }
            double llhX = 0.0;
            double llhY = 0.0;
            int count = 0;
            for (int aa = 0; aa < A; aa++) {
                if (mask[aa][bb]) {
                    double dotprod = billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(y[bb] + x[bb] * dotprod);
                    double prob = score / (1 + score);
                    llhX += (votes[aa][bb] - prob) * dotprod;
                    llhY += votes[aa][bb] - prob;
                    count++;
                }
            }
            x[bb] += (llhX / count - x[bb] / var) * rate;
            y[bb] += (llhY / count - y[bb] / var) * rate;
        }
    }
}
