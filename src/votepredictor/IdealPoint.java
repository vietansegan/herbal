package votepredictor;

import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;

/**
 *
 * @author vietan
 */
public class IdealPoint extends AbstractVotePredictor {

    protected int A; // number of authos
    protected int B; // number of bills
    // params
    protected double[] u; // A
    protected double[] x; // B
    protected double[] y; // B
    // input
    protected int[][] votes;
    protected boolean[][] mask;
    protected ArrayList<Integer> authorIndices;
    protected ArrayList<Integer> billIndices;
    // configure
    protected double alpha;
    protected double eta;
    protected int maxIter;
    // internal
    protected int iter;
    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;
    protected int posAnchor;
    protected int negAnchor;
    protected final double anchorMean = 3.0;
    protected final double anchorVar = 0.01;
    
    public IdealPoint() {
        this.name = "ideal-point";
    }

    public IdealPoint(String name) {
        super(name);
    }

    public void configure() {
        this.configure(5.0, 0.01, 1000);
    }

    public void configure(double alpha, double eta, int maxIter) {
        this.alpha = alpha;
        this.eta = eta;
        this.maxIter = maxIter;
    }

    @Override
    public String getName() {
        return this.name
                + "_a-" + MiscUtils.formatDouble(alpha)
                + "_e-" + MiscUtils.formatDouble(eta)
                + "_m-" + maxIter;
    }

    public double[] getUs() {
        return this.u;
    }

    public double[] getXs() {
        return this.x;
    }

    public double[] getYs() {
        return this.y;
    }

    public ArrayList<String> getAuthorVocab() {
        return this.authorVocab;
    }

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public ArrayList<String> getVoteVocab() {
        return this.voteVocab;
    }

    public void setVoteVocab(ArrayList<String> voteVoc) {
        this.voteVocab = voteVoc;
    }

    public double getLearningRate() {
        return eta * Math.pow(alpha, -(double) iter / maxIter);
    }

    /**
     * Set training data.
     *
     * @param votes
     * @param authorIndices
     * @param billIndices
     * @param trainVotes
     */
    public void setTrain(int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] trainVotes) {
        // list of authors
        this.authorIndices = authorIndices;
        if (authorIndices == null) {
            this.authorIndices = new ArrayList<>();
            for (int aa = 0; aa < votes.length; aa++) {
                this.authorIndices.add(aa);
            }
        }
        this.A = this.authorIndices.size();

        // list of bills
        this.billIndices = billIndices;
        if (billIndices == null) {
            this.billIndices = new ArrayList<>();
            for (int bb = 0; bb < votes[0].length; bb++) {
                this.billIndices.add(bb);
            }
        }
        this.B = this.billIndices.size();

        this.votes = new int[A][B];
        this.mask = new boolean[A][B];
        for (int ii = 0; ii < A; ii++) {
            int aa = this.authorIndices.get(ii);
            for (int jj = 0; jj < B; jj++) {
                int bb = this.billIndices.get(jj);
                this.votes[ii][jj] = votes[aa][bb];
                this.mask[ii][jj] = trainVotes[aa][bb];
            }
        }
    }

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
                this.u[aa] = SamplerUtils.getGaussian(0.0, 5.0);
            }
        }

        this.x = new double[B];
        this.y = new double[B];
        for (int b = 0; b < B; b++) {
            this.x[b] = SamplerUtils.getGaussian(0.0, 5.0);
            this.y[b] = SamplerUtils.getGaussian(0.0, 5.0);
        }
    }

    public void train() {
        initialize(); // initialize
        iterate(); // iterate
    }

    protected void iterate() {
        int stepSize = MiscUtils.getRoundStepSize(maxIter, 10);
        for (iter = 0; iter < maxIter; iter++) {
            if (verbose && iter % stepSize == 0) {
                double avgLlh = getLogLikelihood();
                logln("--- Iter " + iter + " / " + maxIter
                        + "\tllh = " + MiscUtils.formatDouble(avgLlh)
                        + ". positive anchor (" + posAnchor + "): "
                        + MiscUtils.formatDouble(u[posAnchor])
                        + ". negative anchor (" + negAnchor + "): "
                        + MiscUtils.formatDouble(u[negAnchor]));
            }
            updateUs();
            updateXYs();
        }
    }

    protected void updateUs() {
        double aRate = getLearningRate();
        for (int a = 0; a < A; a++) {
            double grad = 0.0;
            for (int b = 0; b < votes[a].length; b++) {
                if (mask[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    grad += x[b] * (votes[a][b] - prob); // only work for 0 and 1
                }
            }
            u[a] += aRate * grad;
        }
    }

    protected void updateXYs() {
        double bRate = getLearningRate();
        for (int b = 0; b < B; b++) {
            double gradX = 0.0;
            double gradY = 0.0;
            for (int a = 0; a < A; a++) {
                if (mask[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    gradX += u[a] * (votes[a][b] - prob);
                    gradY += votes[a][b] - prob;
                }
            }
            x[b] += bRate * gradX;
            y[b] += bRate * gradY;
        }
    }

    public double getLogLikelihood() {
        double llh = 0.0;
        int count = 0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (mask[aa][bb]) {
                    double score = u[aa] * x[bb] + y[bb];
                    llh += votes[aa][bb] * score - Math.log(1 + Math.exp(score));
                    count++;
                }
            }
        }
        return llh / count;
    }

    public SparseVector[] test(boolean[][] testVotes) {
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < predictions.length; aa++) {
            predictions[aa] = new SparseVector();
            for (int bb = 0; bb < testVotes[aa].length; bb++) {
                if (testVotes[aa][bb]) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    double val = score / (1 + score);
                    predictions[aa].set(bb, val);
                }
            }
        }
        return predictions;
    }    

    @Override
    public void output(File modelFile) {
        if (verbose) {
            logln("Outputing model to " + modelFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(modelFile);
            // authors
            writer.write(A + "\n");
            for (int aa = 0; aa < A; aa++) {
                writer.write(u[aa] + "\n");
            }
            // bills
            writer.write(B + "\n");
            for (int bb = 0; bb < B; bb++) {
                writer.write(x[bb] + "\t" + y[bb] + "\n");
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
            // authors
            A = Integer.parseInt(reader.readLine());
            this.u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                u[aa] = Double.parseDouble(reader.readLine());
            }
            // bills
            B = Integer.parseInt(reader.readLine());
            this.x = new double[B];
            this.y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                String[] sline = reader.readLine().split("\t");
                x[bb] = Double.parseDouble(sline[0]);
                y[bb] = Double.parseDouble(sline[1]);
            }
            reader.close();
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from " + modelFile);
        }
    }
}
