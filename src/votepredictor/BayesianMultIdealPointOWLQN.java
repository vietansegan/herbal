package votepredictor;

import edu.stanford.nlp.optimization.DiffFunction;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import optimization.OWLQN;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.SparseVector;

/**
 * Bayesian Multi-dimensional Ideal Point model using OWL-QN for optimization.
 *
 * @author vietan
 */
public class BayesianMultIdealPointOWLQN extends BayesianIdealPoint {

    protected int K;
    protected double l1;
    protected double l2;

    protected double[][] us; // [K]
    protected double[][] xs; // [K+1]: K x's and 1 y

    public BayesianMultIdealPointOWLQN() {
        this.name = "bayesian-mult-ideal-point-owlqn";
    }

    public BayesianMultIdealPointOWLQN(String name) {
        super(name);
    }

    public void configure(int maxIter, double l1, double l2, int K) {
        this.maxIter = maxIter;
        this.l1 = l1;
        this.l2 = l2;
        this.K = K;
    }

    @Override
    public String getName() {
        return this.name
                + "_K-" + K
                + "_m-" + maxIter
                + "_l1-" + MiscUtils.formatDouble(l1)
                + "_l2-" + MiscUtils.formatDouble(l2);
    }

    @Override
    protected void initialize() {
        this.us = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            if (validAs[aa]) {
                for (int kk = 0; kk < K; kk++) {
                    this.us[aa][kk] = SamplerUtils.getGaussian(0.0,
                            l2 != 0 ? 1.0 / Math.sqrt(l2) : 3.0);
                }
            }
        }

        this.xs = new double[B][K + 1];
        for (int bb = 0; bb < B; bb++) {
            if (validBs[bb]) {
                for (int kk = 0; kk < K + 1; kk++) {
                    this.xs[bb][kk] = SamplerUtils.getGaussian(0.0,
                            l2 != 0 ? 1.0 / Math.sqrt(l2) : 3.0);
                }
            }
        }
    }

    @Override
    public SparseVector[] test(boolean[][] testVotes) {
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < predictions.length; aa++) {
            predictions[aa] = new SparseVector();
            for (int bb = 0; bb < testVotes[aa].length; bb++) {
                if (testVotes[aa][bb]) {
                    double dotprod = xs[bb][K]; // y
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double val = score / (1 + score);
                    predictions[aa].set(bb, val);
                }
            }
        }
        return predictions;
    }

    @Override
    protected void iterate() {
        int stepSize = MiscUtils.getRoundStepSize(maxIter, 10);
        for (iter = 0; iter < maxIter; iter++) {
            if (verbose && iter % stepSize == 0) {
                double avgLlh = getLogLikelihood();
                logln("--- Iter " + iter + " / " + maxIter
                        + "\tllh = " + avgLlh);
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
        for (int aa = 0; aa < A; aa++) {
            if (!this.validAs[aa]) {
                continue;
            }
            OWLQN minimizer = new OWLQN();
            minimizer.setQuiet(true);
            minimizer.setMaxIters(100);
            UDiffFunc udiff = new UDiffFunc(aa);
            minimizer.minimize(udiff, us[aa], l1);
        }
    }

    @Override
    protected void updateXYs() {
        for (int bb = 0; bb < B; bb++) {
            if (!this.validBs[bb]) {
                continue;
            }
            OWLQN minimizer = new OWLQN();
            minimizer.setQuiet(true);
            minimizer.setMaxIters(100);
            XYDiffFunc xydiff = new XYDiffFunc(bb);
            minimizer.minimize(xydiff, xs[bb], l1);
        }
    }

    @Override
    public double getLogLikelihood() {
        double llh = 0.0;
        int count = 0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (mask[aa][bb]) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    llh += votes[aa][bb] * dotprod - Math.log(1 + Math.exp(dotprod));
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
                for (int kk = 0; kk < K + 1; kk++) {
                    writer.write(bb + "\t" + kk + "\t" + xs[bb][kk] + "\n");
                }
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
            this.xs = new double[B][K + 1];
            for (int bb = 0; bb < B; bb++) {
                for (int kk = 0; kk < K + 1; kk++) {
                    String[] sline = reader.readLine().split("\t");
                    if (bb != Integer.parseInt(sline[0])
                            || kk != Integer.parseInt(sline[1])) {
                        throw new RuntimeException("Exception");
                    }
                    this.xs[bb][kk] = Double.parseDouble(sline[2]);
                }
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from " + modelFile);
        }
    }

    class UDiffFunc implements DiffFunction {

        private final int aa;

        public UDiffFunc(int aa) {
            this.aa = aa;
        }

        @Override
        public int domainDimension() {
            return K;
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int bb = 0; bb < votes[aa].length; bb++) {
                if (mask[aa][bb]) {
                    double dotprod = xs[bb][K]; // y
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += w[kk] * xs[bb][kk];
                    }
                    llh += votes[aa][bb] * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }

            double val = -llh;
            if (l2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += l2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int bb = 0; bb < votes[aa].length; bb++) {
                if (mask[aa][bb]) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += w[kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] -= xs[bb][kk] * (votes[aa][bb] - prob);
                    }
                }
            }
            if (l2 > 0) {
                for (int kk = 0; kk < w.length; kk++) {
                    grads[kk] += 2 * l2 * w[kk];
                }
            }
            return grads;
        }
    }

    class XYDiffFunc implements DiffFunction {

        private final int bb;

        public XYDiffFunc(int bb) {
            this.bb = bb;
        }

        @Override
        public int domainDimension() {
            return K + 1;
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int aa = 0; aa < votes.length; aa++) {
                if (mask[aa][bb]) {
                    double dotprod = w[K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * w[kk];
                    }
                    llh += votes[aa][bb] * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }

            double val = -llh;
            if (l2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += l2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K + 1];
            for (int aa = 0; aa < votes.length; aa++) {
                if (mask[aa][bb]) {
                    double dotprod = w[K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * w[kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] -= us[aa][kk] * (votes[aa][bb] - prob);
                    }
                    grads[K] -= 1.0 * (votes[aa][bb] - prob);
                }
            }
            if (l2 > 0) {
                for (int kk = 0; kk < w.length; kk++) {
                    grads[kk] += 2 * l2 * w[kk];
                }
            }
            return grads;
        }
    }
}
