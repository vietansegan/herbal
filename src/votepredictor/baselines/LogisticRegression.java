package votepredictor.baselines;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import optimization.OWLQNLogisticRegression;
import optimization.RidgeLogisticRegressionLBFGS;
import util.IOUtils;
import util.MiscUtils;
import util.SamplerUtils;
import util.SparseVector;
import votepredictor.AbstractVotePredictor;

/**
 *
 * @author vietan
 */
public class LogisticRegression extends AbstractVotePredictor {

    public static final double UNDEFINED = -1.0;

    public enum OptType {

        LBFGS, OWLQN
    };
    // input
    protected int V;
    protected int A;
    protected int B;
    protected ArrayList<Integer> authorIndices;
    protected ArrayList<Integer> billIndices;

    // configure
    protected OptType optType;
    protected double mu;
    protected double sigma;
    protected double l1;
    protected double l2;

    protected SparseVector[] authorVectors;                 // A
    protected HashMap<Integer, Integer>[] authorVoteMap;    // A

    protected HashMap<Integer, double[]> weights; // B * V

    public LogisticRegression() {
        this.name = "logreg";
    }

    public LogisticRegression(String bname) {
        super(bname);
    }

    public void configure(int V) {
        this.configure(V, OptType.LBFGS, 0.0, 1.0);
    }

    public void configure(int V, OptType optType, double param1, double param2) {
        this.V = V;
        this.optType = optType;
        if (this.optType == OptType.LBFGS) {
            this.mu = param1;
            this.sigma = param2;
        } else if (this.optType == OptType.OWLQN) {
            this.l1 = param1;
            this.l2 = param2;
        } else {
            throw new RuntimeException("Optimization type " + optType + " not supported");
        }
    }

    @Override
    public String getName() {
        String configName = this.name + "-" + this.optType;
        if (this.optType == OptType.LBFGS) {
            configName += "_m-" + MiscUtils.formatDouble(mu)
                    + "_s-" + MiscUtils.formatDouble(sigma);
        } else if (this.optType == OptType.OWLQN) {
            configName += "_l1-" + MiscUtils.formatDouble(l1)
                    + "_l2-" + MiscUtils.formatDouble(l2);
        } else {
            throw new RuntimeException("Optimization type " + optType + " not supported");
        }
        return configName;
    }

    /**
     * Train a logistic regression model for each bill. Features are normalized
     * frequency of word types in the vocabulary.
     *
     * @param docIndices
     * @param words
     * @param authors
     * @param votes
     * @param authorIndices
     * @param billIndices
     * @param trainVotes
     */
    public void train(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] trainVotes) {
        if (verbose) {
            logln("Setting up training ...");
        }
        // list of training authors
        this.authorIndices = authorIndices;
        if (authorIndices == null) {
            this.authorIndices = new ArrayList<>();
            for (int aa = 0; aa < votes.length; aa++) {
                this.authorIndices.add(aa);
            }
        }
        A = this.authorIndices.size();

        // list of bills
        this.billIndices = billIndices;
        if (billIndices == null) {
            this.billIndices = new ArrayList<>();
            for (int bb = 0; bb < votes[0].length; bb++) {
                this.billIndices.add(bb);
            }
        }
        this.B = this.billIndices.size();

        // training votes
        this.authorVoteMap = new HashMap[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorVoteMap[aa] = new HashMap<>();
            int author = this.authorIndices.get(aa);
            for (int bb : this.billIndices) {
                if (trainVotes[author][bb]) {
                    this.authorVoteMap[aa].put(bb, votes[author][bb]);
                }
            }
        }

        if (docIndices == null) { // add all documents
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                docIndices.add(dd);
            }
        }

        this.authorVectors = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorVectors[aa] = new SparseVector(V);
        }
        for (int dd : docIndices) {
            int author = authors[dd];
            int aa = this.authorIndices.indexOf(author);
            if (aa < 0) {
                continue;
            }
            for (int nn = 0; nn < words[dd].length; nn++) {
                this.authorVectors[aa].change(words[dd][nn], 1.0);
            }
        }
        for (SparseVector authorVec : this.authorVectors) {
            authorVec.normalize();
        }

        // train a logistic regressor for each bill
        if (verbose) {
            logln("Learning parameters ...");
        }
        this.weights = new HashMap<>();
        for (int bb = 0; bb < B; bb++) {
            int bill = this.billIndices.get(bb);
            this.weights.put(bill, trainLogisticRegressor(bb));
        }
    }

    /**
     * Train a logistic regression model for a bill.
     *
     * @param bb Bill index
     * @return Weight vector
     */
    private double[] trainLogisticRegressor(int bb) {
        if (verbose) {
            logln("--- Learning logistic regressor for bill " + bb + " / " + B);
        }
        ArrayList<Integer> labelList = new ArrayList<>();
        ArrayList<SparseVector> authorVecList = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            int bill = this.billIndices.get(bb);
            Integer vote = this.authorVoteMap[aa].get(bill);
            if (vote == null || vote == Vote.MISSING) {
                continue;
            }
            if (vote == Vote.WITH) {
                labelList.add(1);
            } else {
                labelList.add(0);
            }
            authorVecList.add(this.authorVectors[aa]);
        }

        int[] labels = new int[labelList.size()];
        SparseVector[] designMatrix = new SparseVector[authorVecList.size()];
        for (int ii = 0; ii < labelList.size(); ii++) {
            labels[ii] = labelList.get(ii);
            designMatrix[ii] = authorVecList.get(ii);
        }

        double[] ws = new double[V];
        for (int vv = 0; vv < V; vv++) {
            ws[vv] = SamplerUtils.getGaussian(mu, sigma);
        }

        if (designMatrix.length == 0) {
            System.out.println("Skipping bill " + bb);
            return null;
        }

        if (this.optType == OptType.LBFGS) {
            RidgeLogisticRegressionLBFGS logreg = new RidgeLogisticRegressionLBFGS(
                    labels, ws, designMatrix, mu, sigma);
            LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(logreg);
            boolean converged = false;
            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            if (verbose) {
                logln("--- --- converged? " + converged);
            }

            // update regression parameters
            ws = new double[V];
            for (int vv = 0; vv < V; vv++) {
                ws[vv] = logreg.getParameter(vv);
            }
        } else if (this.optType == OptType.OWLQN) {
            OWLQNLogisticRegression logreg = new OWLQNLogisticRegression(name, l1, l2);
            logreg.train(designMatrix, labels, V);

            ws = new double[V]; // update regression parameters
            System.arraycopy(logreg.getWeights(), 0, ws, 0, V);
        }

        return ws;
    }

    public SparseVector[] test(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            ArrayList<Integer> authorIndices,
            boolean[][] testVotes) {
        if (authorIndices == null) {
            throw new RuntimeException("List of test authors is null");
        }
        int testA = authorIndices.size();
        SparseVector[] testAuthorVecs = new SparseVector[testA];
        for (int aa = 0; aa < testA; aa++) {
            testAuthorVecs[aa] = new SparseVector(this.V);
        }
        if (docIndices == null) { // add all documents
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                docIndices.add(dd);
            }
        }
        for (int dd : docIndices) {
            int author = authors[dd];
            int aa = authorIndices.indexOf(author);
            if (aa < 0) {
                continue;
            }
            for (int nn = 0; nn < words[dd].length; nn++) {
                testAuthorVecs[aa].change(words[dd][nn], 1.0);
            }
        }

        for (SparseVector testAuthorVec : testAuthorVecs) {
            testAuthorVec.normalize();
        }

        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < testA; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(testVotes[author].length);
            for (int bb = 0; bb < testVotes[author].length; bb++) {
                if (testVotes[author][bb]) {
                    double val = Math.exp(testAuthorVecs[aa].dotProduct(weights.get(bb)));
                    predictions[author].set(bb, val / (1.0 + val));
                }
            }
        }
        return predictions;
    }

    @Override
    public void input(File modelFile) {
        if (verbose) {
            logln("Inputing model from " + modelFile);
        }
        try {
            this.weights = new HashMap<>();
            BufferedReader reader = IOUtils.getBufferedReader(modelFile);
            String line;
            String[] sline;
            while ((line = reader.readLine()) != null) {
                sline = line.split("\t");
                int bill = Integer.parseInt(sline[0]);
                double[] ws = null;
                if (!sline[1].equals("null")) {
                    ws = MiscUtils.stringToDoubleArray(sline[1]);
                }
                this.weights.put(bill, ws);
            }
            reader.close();
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from " + modelFile);
        }
    }

    @Override
    public void output(File modelFile) {
        if (verbose) {
            logln("Outputing model to " + modelFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(modelFile);
            for (int bb : this.weights.keySet()) {
                if (this.weights.get(bb) == null) {
                    writer.write(bb + "\tnull\n");
                } else {
                    writer.write(bb + "\t" + MiscUtils.arrayToString(this.weights.get(bb)) + "\n");
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing model to " + modelFile);
        }
    }
}
