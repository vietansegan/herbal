package votepredictor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;

/**
 * Bayesian Multi-dimensional Ideal Point model using gradient descent.
 *
 * @author vietan
 */
public class BayesianMultIdealPoint extends BayesianIdealPoint {

    protected int K;

    protected double[][] us;
    protected double[][] xs;

    public BayesianMultIdealPoint() {
        this.name = "bayesian-mult-ideal-point";
    }

    public BayesianMultIdealPoint(String name) {
        super(name);
    }

    @Override
    public void configure() {
        this.configure(5.0, 0.01, 1000, 0.0, 2.5, 5);
    }

    @Override
    public void configure(double alpha, double eta, int maxIter) {
        this.configure(alpha, eta, maxIter, 0.0, 2.5, 5);
    }

    @Override
    public void configure(double alpha, double eta, int maxIter,
            double mu, double sigma) {
        this.configure(alpha, eta, maxIter, mu, sigma, 5);
    }

    public void configure(double alpha, double eta, int maxIter,
            double mu, double sigma, int K) {
        super.configure(alpha, eta, maxIter, mu, sigma);
        this.K = K;
    }

    @Override
    public String getName() {
        return this.name
                + "_K-" + K
                + "_a-" + MiscUtils.formatDouble(alpha)
                + "_e-" + MiscUtils.formatDouble(eta)
                + "_m-" + maxIter
                + "_m-" + MiscUtils.formatDouble(mean)
                + "_s-" + MiscUtils.formatDouble(var);
    }

    @Override
    public SparseVector[] test(boolean[][] testVotes) {
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < predictions.length; aa++) {
            predictions[aa] = new SparseVector();
            for (int bb = 0; bb < testVotes[aa].length; bb++) {
                if (testVotes[aa][bb]) {
                    double score = Math.exp(StatUtils.dotProduct(us[aa], xs[bb]) + y[bb]);
                    double val = score / (1 + score);
                    predictions[aa].set(bb, val);
                }
            }
        }
        return predictions;
    }

    @Override
    protected void initialize() {
        this.us = new double[A][K];
        for (int a = 0; a < A; a++) {
            for (int k = 0; k < K; k++) {
                this.us[a][k] = SamplerUtils.getGaussian(mean, var);
            }
        }

        this.xs = new double[B][K];
        this.y = new double[B];
        for (int b = 0; b < B; b++) {
            this.y[b] = SamplerUtils.getGaussian(mean, var);
            for (int k = 0; k < K; k++) {
                this.xs[b][k] = SamplerUtils.getGaussian(mean, var);
            }
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
        for (int aa = 0; aa < A; aa++) {
            double[] grads = new double[K];
            for (int bb = 0; bb < votes[aa].length; bb++) {
                if (mask[aa][bb]) {
                    double score = Math.exp(StatUtils.dotProduct(us[aa], xs[bb]) + y[bb]);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] += xs[bb][kk] * (votes[aa][bb] - prob); // only work for 0 and 1
                    }
                }
            }
            for (int kk = 0; kk < K; kk++) {
                grads[kk] -= (us[aa][kk] - mean) / var;
                us[aa][kk] += aRate * grads[kk];
            }
        }
    }

    @Override
    protected void updateXYs() {
        double bRate = getLearningRate();
        for (int bb = 0; bb < B; bb++) {
            double[] gradXs = new double[K];
            double gradY = 0.0;
            for (int aa = 0; aa < A; aa++) {
                if (mask[aa][bb]) {
                    double score = Math.exp(StatUtils.dotProduct(us[aa], xs[bb]) + y[bb]);
                    double prob = score / (1 + score);
                    gradY += votes[aa][bb] - prob;
                    for (int kk = 0; kk < K; kk++) {
                        gradXs[kk] += us[aa][kk] * (votes[aa][bb] - prob);
                    }
                }
            }

            // prior
            for (int kk = 0; kk < K; kk++) {
                gradXs[kk] -= (xs[bb][kk] - mean) / var;
            }
            gradY -= (y[bb] - mean) / var;

            y[bb] += bRate * gradY;
            for (int kk = 0; kk < K; kk++) {
                xs[bb][kk] += bRate * gradXs[kk];
            }
        }
    }

    @Override
    public double getLogLikelihood() {
        double llh = 0.0;
        int count = 0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (mask[aa][bb]) {
                    double score = StatUtils.dotProduct(us[aa], xs[bb]) + y[bb];
                    llh += votes[aa][bb] * score - Math.log(1 + Math.exp(score));
                    count++;
                }
            }
        }
        return llh / count;
    }

    @Override
    public void output(File modelFile) {
        if (verbose) {
            logln("Outputing model to " + modelFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(modelFile);
            writer.write(K + "\n");
            // authors
            writer.write(A + "\n");
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    writer.write(aa + "\t" + kk + "\t" + us[aa][kk] + "\n");
                }
            }
            // bills
            writer.write(B + "\n");
            for (int bb = 0; bb < B; bb++) {
                for (int kk = 0; kk < K; kk++) {
                    writer.write(bb + "\t" + kk + "\t" + xs[bb][kk] + "\n");
                }
            }
            for (int bb = 0; bb < B; bb++) {
                writer.write(bb + "\t" + y[bb] + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing model to " + modelFile);
        }
    }

    @Override
    public void input(File modelFile) {
        if (verbose) {
            logln("Inputing model from " + modelFile);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(modelFile);
            K = Integer.parseInt(reader.readLine());
            // authors
            A = Integer.parseInt(reader.readLine());
            this.us = new double[A][K];
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    String[] sline = reader.readLine().split("\t");
                    if (aa != Integer.parseInt(sline[0])
                            || kk != Integer.parseInt(sline[1])) {
                        throw new RuntimeException("Exception");
                    }
                    this.us[aa][kk] = Double.parseDouble(sline[2]);
                }
            }
            // bills
            B = Integer.parseInt(reader.readLine());
            this.xs = new double[B][K];
            for (int bb = 0; bb < B; bb++) {
                for (int kk = 0; kk < K; kk++) {
                    String[] sline = reader.readLine().split("\t");
                    if (bb != Integer.parseInt(sline[0])
                            || kk != Integer.parseInt(sline[1])) {
                        throw new RuntimeException("Exception");
                    }
                    this.xs[bb][kk] = Double.parseDouble(sline[2]);
                }
            }

            this.y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                String[] sline = reader.readLine().split("\t");
                if (bb != Integer.parseInt(sline[0])) {
                    throw new RuntimeException("Exception");
                }
                y[bb] = Double.parseDouble(sline[1]);
            }
            reader.close();
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from " + modelFile);
        }
    }
}
