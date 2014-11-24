package votepredictor.baselines;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Vote;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import de.bwaldvogel.liblinear.Model;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import optimization.OWLQNLogisticRegression;
import optimization.RidgeLogisticRegressionLBFGS;
import sampling.util.SparseCount;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.normalizer.AbstractNormalizer;
import util.normalizer.MinMaxNormalizer;
import util.normalizer.ZNormalizer;
import votepredictor.AbstractVotePredictor;

/**
 *
 * @author vietan
 */
public class LogisticRegression extends AbstractVotePredictor {

    public static final double UNDEFINED = -1.0;

    public enum OptType {

        LBFGS, OWLQN, LIBLINEAR
    };

    public enum NormalizeType {

        MINMAX, ZSCORE, TFIDF, NONE
    };
    // input
    protected int V;
    protected int A;
    protected int B;
    protected ArrayList<Integer> authorIndices;
    protected ArrayList<Integer> billIndices;

    // configure
    protected OptType optType;
    protected NormalizeType normType;
    // L-BFGS
    protected double mu;
    protected double sigma;
    // OWL-QN
    protected double l1;
    protected double l2;
    // LIBLINEAR
    protected double c;
    protected double epsilon;
    protected Model[] models;

    protected SparseVector[] authorVectors;                 // A
    protected HashMap<Integer, Integer>[] authorVoteMap;    // A

    protected HashMap<Integer, double[]> weights; // [B] * [number of features]
    protected AbstractNormalizer[] normalizers;
    protected double[] idfs;

    public LogisticRegression() {
        this.name = "logreg";
    }

    public LogisticRegression(String bname) {
        super(bname);
    }

    public OptType getOptType() {
        return this.optType;
    }

    public void configure(int V) {
        this.configure(V, OptType.LBFGS, NormalizeType.MINMAX, 0.0, 1.0);
    }

    public void configure(int V, OptType optType, NormalizeType normType,
            double param1, double param2) {
        this.V = V;
        this.optType = optType;
        this.normType = normType;
        if (this.optType == OptType.LBFGS) {
            this.mu = param1;
            this.sigma = param2;
        } else if (this.optType == OptType.OWLQN) {
            this.l1 = param1;
            this.l2 = param2;
        } else if (this.optType == OptType.LIBLINEAR) {
            this.c = param1;
            this.epsilon = param2;
        } else {
            throw new RuntimeException("Optimization type " + optType + " not supported");
        }

        if (verbose) {
            logln("Configured.");
            logln("--- Opt type: " + this.optType);
            logln("--- Param 1: " + param1);
            logln("--- Param 2: " + param2);
        }
    }

    @Override
    public String getName() {
        String configName = this.name + "-" + this.optType
                + "-" + this.normType;
        if (this.optType == OptType.LBFGS) {
            configName += "_m-" + MiscUtils.formatDouble(mu)
                    + "_s-" + MiscUtils.formatDouble(sigma);
        } else if (this.optType == OptType.OWLQN) {
            configName += "_l1-" + MiscUtils.formatDouble(l1)
                    + "_l2-" + MiscUtils.formatDouble(l2);
        } else if (this.optType == OptType.LIBLINEAR) {
            configName += "_c-" + MiscUtils.formatDouble(c)
                    + "_e-" + MiscUtils.formatDouble(epsilon);
        } else {
            throw new RuntimeException("Optimization type " + optType + " not supported");
        }
        return configName;
    }

    public HashMap<Integer, double[]> getWeights() {
        return this.weights;
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
     * @param addFeatures Additional feature vector for each author
     * @param Fs
     */
    public void train(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] trainVotes,
            ArrayList<SparseVector[]> addFeatures,
            ArrayList<Integer> Fs) {
        if (verbose) {
            logln("Setting up training ...");
        }

        if (Fs == null || addFeatures == null) {
            throw new RuntimeException("Additional features can be empty but not null");
        }

        if (Fs.size() != addFeatures.size()) {
            throw new MismatchRuntimeException(addFeatures.size(), Fs.size());
        }

        int totalF = 0;
        for (int F : Fs) {
            totalF += F;
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

        if (normType == NormalizeType.TFIDF) {
            this.idfs = MiscUtils.getIDFs(words, docIndices, V);
        }

        this.authorVectors = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorVectors[aa] = new SparseVector(V + totalF);
        }
        // lexical features
        for (int dd : docIndices) {
            int author = authors[dd];
            int aa = this.authorIndices.indexOf(author);
            if (aa < 0) {
                continue;
            }
            SparseCount counts = new SparseCount();
            for (int nn = 0; nn < words[dd].length; nn++) {
                counts.increment(words[dd][nn]);
            }
            for (int vv : counts.getIndices()) {
                int count = counts.getCount(vv);
                if (normType == NormalizeType.TFIDF) {
                    double tfidf = Math.log(count + 1) * idfs[vv];
                    this.authorVectors[aa].change(vv, tfidf);
                } else {
                    this.authorVectors[aa].change(vv, count);
                }
            }
        }
        if (normType != NormalizeType.TFIDF) {
            for (SparseVector authorVec : this.authorVectors) { // normalize raw counts
                authorVec.normalize();
            }
        }

        // additional features
        if (totalF != 0) {
            int startIdx = V;
            for (int ii = 0; ii < Fs.size(); ii++) {
                for (int aa = 0; aa < A; aa++) {
                    SparseVector authorAddFeatures = addFeatures.get(ii)[aa];
                    for (int idx : authorAddFeatures.getIndices()) {
                        authorVectors[aa].set(startIdx + idx, authorAddFeatures.get(idx));
                    }
                }
                startIdx += Fs.get(ii);
            }
        }

        if (normType == NormalizeType.MINMAX) {
            normalizers = StatUtils.minmaxNormalizeTrainingData(authorVectors, V + totalF);
        } else if (normType == NormalizeType.ZSCORE) {
            normalizers = StatUtils.zNormalizeTrainingData(authorVectors, V + totalF);
        } else if (normType == NormalizeType.NONE) {
            normalizers = null;
        } else if (normType == NormalizeType.TFIDF) {
            normalizers = null;
        } else {
            throw new RuntimeException("Normalization type " + normType
                    + " is not supported");
        }

        // train a logistic regressor for each bill
        if (verbose) {
            logln("Learning parameters ...");
        }
        if (this.optType == OptType.LIBLINEAR) {
            this.models = new Model[B];
            for (int bb = 0; bb < B; bb++) {
                trainLogisticRegressor(bb, V + totalF);
            }
        } else {
            this.weights = new HashMap<>();
            for (int bb = 0; bb < B; bb++) {
                int bill = this.billIndices.get(bb);
                this.weights.put(bill, trainLogisticRegressor(bb, V + totalF));
            }
        }
    }

    /**
     * Train a logistic regression model for a bill.
     *
     * @param bb Bill index
     * @return Weight vector
     */
    private double[] trainLogisticRegressor(int bb, int numFeatures) {
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
            } else if (vote == Vote.AGAINST) {
                labelList.add(0);
            } else {
                throw new RuntimeException("Vote " + vote + " is invalid");
            }
            authorVecList.add(this.authorVectors[aa]);
        }

        int[] labels = new int[labelList.size()];
        SparseVector[] designMatrix = new SparseVector[authorVecList.size()];
        for (int ii = 0; ii < labelList.size(); ii++) {
            labels[ii] = labelList.get(ii);
            designMatrix[ii] = authorVecList.get(ii);
        }

        double[] ws = new double[numFeatures];
        for (int vv = 0; vv < numFeatures; vv++) {
            ws[vv] = SamplerUtils.getGaussian(mu, sigma);
        }

        if (designMatrix.length == 0) {
            System.out.println("Skipping empty bill " + bb);
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
            ws = new double[numFeatures];
            for (int vv = 0; vv < numFeatures; vv++) {
                ws[vv] = logreg.getParameter(vv);
            }
        } else if (this.optType == OptType.OWLQN) {
            OWLQNLogisticRegression logreg = new OWLQNLogisticRegression(name, l1, l2);
            logreg.train(designMatrix, labels, numFeatures);
            ws = new double[numFeatures]; // update regression parameters
            System.arraycopy(logreg.getWeights(), 0, ws, 0, numFeatures);
        } else if (this.optType == OptType.LIBLINEAR) {
            Problem problem = new Problem();
            problem.l = designMatrix.length; // number of training examples
            problem.n = numFeatures;
            problem.x = new FeatureNode[designMatrix.length][numFeatures];
            for (int a = 0; a < designMatrix.length; a++) {
                for (int v = 0; v < numFeatures; v++) {
                    problem.x[a][v] = new FeatureNode(v + 1, designMatrix[a].get(v));
                }
            }
            for (Feature[] nodes : problem.x) {
                int indexBefore = 0;
                for (Feature n : nodes) {
                    if (n.getIndex() <= indexBefore) {
                        throw new IllegalArgumentException("Hello: feature nodes "
                                + "must be sorted by index in ascending order. "
                                + indexBefore + " vs. " + n.getIndex()
                                + "\t" + nodes.length);
                    }
                    indexBefore = n.getIndex();
                }
            }
            double[] dLabels = new double[labels.length];
            for (int ii = 0; ii < labels.length; ii++) {
                dLabels[ii] = labels[ii];
            }
            problem.y = dLabels;
            Parameter parameter = new Parameter(SolverType.L2R_LR, c, epsilon);
            this.models[bb] = Linear.train(problem, parameter);
        }

        return ws;
    }

    /**
     * Make predictions on test data.
     *
     * @param docIndices
     * @param words
     * @param authors
     * @param authorIndices
     * @param testVotes
     * @param addFeatures Additional feature vector for each author
     * @param Fs
     * @return
     */
    public SparseVector[] test(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            ArrayList<Integer> authorIndices,
            boolean[][] testVotes,
            ArrayList<SparseVector[]> addFeatures,
            ArrayList<Integer> Fs) {
        if (authorIndices == null) {
            throw new RuntimeException("List of test authors is null");
        }

        if (Fs == null || addFeatures == null) {
            throw new RuntimeException("Additional features can be empty but not null");
        }

        if (Fs.size() != addFeatures.size()) {
            throw new MismatchRuntimeException(addFeatures.size(), Fs.size());
        }

        int totalF = 0;
        for (int F : Fs) {
            totalF += F;
        }

        int testA = authorIndices.size();
        SparseVector[] testAuthorVecs = new SparseVector[testA];
        for (int aa = 0; aa < testA; aa++) {
            testAuthorVecs[aa] = new SparseVector(V + totalF);
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
            SparseCount counts = new SparseCount();
            for (int nn = 0; nn < words[dd].length; nn++) {
                counts.increment(words[dd][nn]);
            }
            for (int vv : counts.getIndices()) {
                int count = counts.getCount(vv);
                if (normType == NormalizeType.TFIDF) {
                    double tfidf = Math.log(count + 1) * idfs[vv];
                    testAuthorVecs[aa].change(vv, tfidf);
                } else {
                    testAuthorVecs[aa].change(vv, count);
                }
            }
        }

        if (normType != NormalizeType.TFIDF) {
            for (SparseVector testAuthorVec : testAuthorVecs) {
                testAuthorVec.normalize();
            }
        }

        // additional features
        if (totalF != 0) {
            int startIdx = V;
            for (int ii = 0; ii < Fs.size(); ii++) {
                for (int aa = 0; aa < testA; aa++) {
                    SparseVector authorAddFeatures = addFeatures.get(ii)[aa];
                    for (int idx : authorAddFeatures.getIndices()) {
                        testAuthorVecs[aa].set(startIdx + idx, authorAddFeatures.get(idx));
                    }
                }
                startIdx += Fs.get(ii);
            }
        }

        if (normType == NormalizeType.MINMAX || normType == NormalizeType.ZSCORE) {
            StatUtils.normalizeTestData(testAuthorVecs, normalizers);
        } else if (normType == NormalizeType.TFIDF) {
            if (normalizers != null) {
                throw new RuntimeException();
            }
        } else if (normType == NormalizeType.NONE) {
            if (normalizers != null) {
                throw new RuntimeException();
            }
        } else {
            throw new RuntimeException("Normalization type " + normType
                    + " is not supported");
        }

        SparseVector[] predictions = new SparseVector[testVotes.length];
        if (this.optType == OptType.LIBLINEAR) {
            for (int aa = 0; aa < testA; aa++) {
                int author = authorIndices.get(aa);
                predictions[author] = new SparseVector(testVotes[author].length);
            }

            for (int bb = 0; bb < models.length; bb++) {
                for (int aa = 0; aa < testA; aa++) {
                    if (testVotes[authorIndices.get(aa)][bb]) {
                        Feature[] instance = new Feature[testAuthorVecs[aa].getDimension()];
                        for (int v = 0; v < testAuthorVecs[aa].getDimension(); v++) {
                            instance[v] = new FeatureNode(v + 1, testAuthorVecs[aa].get(v));
                        }
                        double predVal = Linear.predict(models[bb], instance);
                        predictions[authorIndices.get(aa)].set(bb, predVal);
                    }
                }
            }
        } else {
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
        }

        return predictions;
    }

    @Override
    public void input(File modelPath) {
        if (verbose) {
            logln("Inputing model from " + modelPath);
        }

        try {
            if (this.optType == OptType.LIBLINEAR) {
                String[] filenames = modelPath.list();
                this.models = new Model[filenames.length];
                for (int bb = 0; bb < this.models.length; bb++) {
                    this.models[bb] = Linear.loadModel(new File(modelPath, bb + ".model"));
                }
            } else {
                this.weights = new HashMap<>();
                BufferedReader reader = IOUtils.getBufferedReader(modelPath);
                String[] sline;
                int numBills = Integer.parseInt(reader.readLine());
                for (int bb = 0; bb < numBills; bb++) {
                    sline = reader.readLine().split("\t");
                    int bill = Integer.parseInt(sline[0]);
                    double[] ws = null;
                    if (!sline[1].equals("null")) {
                        ws = MiscUtils.stringToDoubleArray(sline[1]);
                    }
                    this.weights.put(bill, ws);
                }

                if (normType != NormalizeType.NONE) {
                    if (normType == NormalizeType.TFIDF) {
                        int vocabSize = Integer.parseInt(reader.readLine());
                        if (V != vocabSize) {
                            throw new MismatchRuntimeException(vocabSize, V);
                        }
                        this.idfs = new double[V];
                        for (int vv = 0; vv < V; vv++) {
                            this.idfs[vv] = Double.parseDouble(reader.readLine());
                        }
                    } else {
                        int numFeatures = Integer.parseInt(reader.readLine());
                        this.normalizers = new AbstractNormalizer[numFeatures];
                        for (int ff = 0; ff < numFeatures; ff++) {
                            if (normType == NormalizeType.MINMAX) {
                                normalizers[ff] = MinMaxNormalizer.input(reader.readLine());
                            } else if (normType == NormalizeType.ZSCORE) {
                                normalizers[ff] = ZNormalizer.input(reader.readLine());
                            } else {
                                throw new RuntimeException("Normalization type " + normType
                                        + " is not supported");
                            }
                        }
                    }
                }
                reader.close();
            }
        } catch (IOException | RuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from " + modelPath);
        }
    }

    @Override
    public void output(File modelPath) {
        if (verbose) {
            logln("Outputing model to " + modelPath);
        }
        try {
            if (this.optType == OptType.LIBLINEAR) {
                for (int bb = 0; bb < models.length; bb++) {
                    this.models[bb].save(new File(modelPath, bb + ".model"));
                }
            } else {
                BufferedWriter writer = IOUtils.getBufferedWriter(modelPath);
                writer.write(this.weights.size() + "\n");
                for (int bb : this.weights.keySet()) {
                    if (this.weights.get(bb) == null) {
                        writer.write(bb + "\tnull\n");
                    } else {
                        writer.write(bb + "\t" + MiscUtils.arrayToString(this.weights.get(bb)) + "\n");
                    }
                }

                if (normType != NormalizeType.NONE) {
                    if (normType == NormalizeType.TFIDF) {
                        writer.write(V + "\n");
                        for (double idf : idfs) {
                            writer.write(idf + "\n");
                        }
                    } else {
                        writer.write(normalizers.length + "\n");
                        for (AbstractNormalizer normalizer : normalizers) {
                            if (normType == NormalizeType.MINMAX) {
                                writer.write(MinMaxNormalizer.output((MinMaxNormalizer) normalizer) + "\n");
                            } else if (normType == NormalizeType.ZSCORE) {
                                writer.write(ZNormalizer.output((ZNormalizer) normalizer) + "\n");
                            } else {
                                throw new RuntimeException("Normalization type " + normType
                                        + " is not supported");
                            }
                        }
                    }
                }
                writer.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing model to " + modelPath);
        }
    }
}
