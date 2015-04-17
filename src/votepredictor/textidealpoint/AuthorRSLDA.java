package votepredictor.textidealpoint;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import optimization.RidgeLinearRegressionOptimizable;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.evaluation.RegressionEvaluation;

/**
 *
 * @author vietan
 */
public class AuthorRSLDA extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public static final int INVALID_TOPIC = -1;
    protected double rho;
    protected double mu;
    protected double sigma;

    // inputs
    protected int[][] words; // original documents
    protected boolean[][] validTokens;
    protected int[] authors;
    protected double[] responses; // author responses

    protected int K;
    protected int V;
    protected int A;

    // derive
    protected int D;
    // latent variables
    protected int[][] z;
    protected DirMult[] docTopics;
    protected DirMult[] topicWords;
    protected double[] eta;

    protected double[] authorMeans;
    protected int[] authorTokenCounts;
    private double sqrtRho;

    public AuthorRSLDA() {
        this.basename = "AuthorRSLDA";
    }

    public AuthorRSLDA(String bname) {
        this.basename = bname;
    }

    public void configure(
            String folder,
            int V, int K, int A,
            double alpha,
            double beta,
            double rho, // variance of Gaussian for document observations
            double mu, // mean of Gaussian for regression parameters
            double sigma, // variance of Gaussian for regression parameters
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.V = V;
        this.A = A;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);

        this.rho = rho;
        this.mu = mu;
        this.sigma = sigma;
        this.sqrtRho = Math.sqrt(this.rho);

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.setName();

        this.report = true;

        if (verbose && folder != null) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- vocab size:\t" + V);
            logln("--- number of authors:\t" + A);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta: \t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- response rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- topic mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- topic sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        this.name = this.prefix
                + "_" + this.basename
                + "_K-" + K
                + "_B-" + BURN_IN
                + "_M-" + MAX_ITER
                + "_L-" + LAG
                + "_a-" + formatter.format(hyperparams.get(ALPHA))
                + "_b-" + formatter.format(hyperparams.get(BETA))
                + "_r-" + formatter.format(rho)
                + "_m-" + formatter.format(mu)
                + "_s-" + formatter.format(sigma)
                + "_opt-" + this.paramOptimized;
    }

    public int[][] getZs() {
        return this.z;
    }

    public double[] getEtas() {
        return this.eta;
    }

    public boolean getValidToken(int dd, int nn) {
        return this.validTokens[dd][nn];
    }

    protected void evaluate() {
        RegressionEvaluation eval = new RegressionEvaluation(responses, authorMeans);
        eval.computeCorrelationCoefficient();
        eval.computeMeanSquareError();
        eval.computeMeanAbsoluteError();
        eval.computeRSquared();
        eval.computePredictiveRSquared();
        ArrayList<Measurement> measurements = eval.getMeasurements();
        for (Measurement measurement : measurements) {
            logln("--- --- " + measurement.getName() + ":\t" + measurement.getValue());
        }
    }

    /**
     * Set up data.
     *
     * @param docWords All documents
     * @param docIndices Indices of documents under consideration
     * @param docResponses Responses of all documents, null if test data
     */
    public void setupData(int[][] docWords, boolean[][] docValidTokens,
            int[] docAuthors, double[] authorResponses) {
        this.D = docWords.length;
        this.words = docWords;
        this.authors = docAuthors;
        this.validTokens = docValidTokens;
        this.responses = authorResponses;

        this.numTokens = 0;
        this.authorTokenCounts = new int[A];
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                if (validTokens[dd][nn]) {
                    this.numTokens++;
                    this.authorTokenCounts[aa]++;
                }
            }
        }

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # tokens:\t" + numTokens);
            if (responses != null) {
                logln("--- responses:");
                logln("--- --- mean\t" + MiscUtils.formatDouble(StatUtils.mean(responses)));
                logln("--- --- stdv\t" + MiscUtils.formatDouble(StatUtils.standardDeviation(responses)));
                int[] histogram = StatUtils.bin(responses, 10);
                for (int ii = 0; ii < histogram.length; ii++) {
                    logln("--- --- " + ii + "\t" + histogram[ii]);
                }
            }
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        System.out.println("\n\n");
        logln("Iter = " + iter + " / " + MAX_ITER);
        this.getLogLikelihood();
        this.evaluate();
        return str.toString();
    }

    @Override
    public void initialize() {
        initialize(null);
    }

    public void initialize(double[][] seededTopics) {
        if (verbose) {
            logln("Initializing ...");
        }

        iter = INIT;
        isReporting = true;
        initializeModelStructure(seededTopics);
        initializeDataStructure();
        initializeAssignments();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \n" + getCurrentState());
            getLogLikelihood();
        }
    }

    protected void initializeModelStructure(double[][] seededTopics) {
        if (seededTopics != null && seededTopics.length != K) {
            throw new RuntimeException("Mismatch" + ". K = " + K
                    + ". # prior topics = " + seededTopics.length);
        }

        topicWords = new DirMult[K];
        for (int k = 0; k < K; k++) {
            if (seededTopics != null) {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, seededTopics[k]);
            } else {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }

        eta = new double[K];
        for (int k = 0; k < K; k++) {
            eta[k] = SamplerUtils.getGaussian(mu, sigma);
        }
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int dd = 0; dd < D; dd++) {
            z[dd] = new int[words[dd].length];
            for (int nn = 0; nn < words[dd].length; nn++) {
                if (!validTokens[dd][nn]) {
                    z[dd][nn] = INVALID_TOPIC;
                }
            }
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        authorMeans = new double[A];
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization " + initState + " not supported");
        }
    }

    protected void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
    }

    @Override
    public void iterate() {
        updateEtas(); // update topic regression parameters
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED); // sample token assignments
    }

    /**
     * Sample topic assignment for each token.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe
     * @return Elapsed time
     */
    protected long sampleZs(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe) {
        if (isReporting) {
            logln("+++ Sampling assignments ...");
        }
        long sTime = System.currentTimeMillis();
        numTokensChanged = 0;
        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int n = 0; n < words[d].length; n++) {
                if (!validTokens[d][n]) {
                    continue;
                }

                if (removeFromModel) {
                    topicWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][n]);
                    authorMeans[aa] -= eta[z[d][n]] / authorTokenCounts[aa];
                }

                double[] logprobs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    logprobs[kk]
                            = Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[kk].getProbability(words[d][n]));
                    if (observe) {
                        double aMean = authorMeans[aa] + eta[kk] / authorTokenCounts[aa];
                        double resLLh = StatUtils.logNormalProbability(responses[aa], aMean, Math.sqrt(rho));
                        logprobs[kk] += resLLh;
                    }
                }
                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
                if (sampledZ == K) {
                    logln("iter = " + iter + ". d = " + d + ". n = " + n);
                    for (int kk = 0; kk < K; kk++) {
                        logln("k = " + kk
                                + ". " + (Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA)))
                                + ". " + (Math.log(topicWords[kk].getProbability(words[d][n]))));
                    }
                    throw new RuntimeException("Out-of-bound sample. "
                            + "SampledZ = " + sampledZ);
                }

                if (z[d][n] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[d][n] = sampledZ;

                if (addToModel) {
                    topicWords[z[d][n]].increment(words[d][n]);
                }
                if (addToData) {
                    docTopics[d].increment(z[d][n]);
                    authorMeans[aa] += eta[z[d][n]] / authorTokenCounts[aa];
                }
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # tokens changed: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")");
        }
        return eTime;
    }

    /**
     * Optimize etas using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();

        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(K);
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int kk : docTopics[dd].getSparseCounts().getIndices()) {
                int count = docTopics[dd].getCount(kk);
                designMatrix[aa].change(kk, (double) count / authorTokenCounts[aa]);
            }
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, eta, designMatrix, rho, mu, sigma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (isReporting) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int kk = 0; kk < K; kk++) {
            eta[kk] = optimizable.getParameter(kk);
        }
        // update author means
        for (int aa = 0; aa < A; aa++) {
            authorMeans[aa] = 0.0;
            for (int kk : designMatrix[aa].getIndices()) {
                authorMeans[aa] += designMatrix[aa].get(kk) * eta[kk];
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    @Override
    public double getLogLikelihood() {
        return 0.0;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        return 0.0;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {

    }

    @Override
    public void validate(String msg) {

    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        try {
            // model
            StringBuilder modelStr = new StringBuilder();
            for (int kk = 0; kk < K; kk++) {
                modelStr.append(kk).append("\n");
                modelStr.append(eta[kk]).append("\n");
                modelStr.append(DirMult.output(topicWords[kk])).append("\n");
            }

            // assignments
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");
                for (int n = 0; n < z[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);

            this.outputZipFile(filepath, contentStrs, entryFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while outputing state to "
                    + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }

        try {
            inputModel(filepath);
            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while inputing state from "
                    + filepath);
        }

        validate("Done reading state from " + filepath);
    }

    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + ModelFileExt);
            topicWords = new DirMult[K];
            eta = new double[K];
            for (int kk = 0; kk < K; kk++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != kk) {
                    throw new RuntimeException("Indices mismatch when loading model");
                }
                eta[kk] = Double.parseDouble(reader.readLine());
                topicWords[kk] = DirMult.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    public void inputAssignments(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }

        try {
            // initialize
            this.initializeDataStructure();

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + AssignmentFileExt);
            for (int d = 0; d < D; d++) {
                int docIdx = Integer.parseInt(reader.readLine().split("\t")[0]);
                if (docIdx != d) {
                    throw new RuntimeException("Indices mismatch when loading assignments");
                }
                docTopics[d] = DirMult.input(reader.readLine());

                String[] sline = reader.readLine().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("[MISMATCH]. Doc "
                            + d + ". " + sline.length + " vs. " + words[d].length);
                }
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        this.outputTopicTopWords(file, numTopWords, null);
    }

    public void outputTopicTopWords(File file, int numTopWords, ArrayList<String> topicLabels) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        ArrayList<RankingItem<Integer>> sortedTopics = new ArrayList<RankingItem<Integer>>();
        for (int k = 0; k < K; k++) {
            sortedTopics.add(new RankingItem<Integer>(k, eta[k]));
        }
        Collections.sort(sortedTopics);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int ii = 0; ii < K; ii++) {
                int k = sortedTopics.get(ii).getObject();
                String topicLabel = "Topic " + k;
                if (topicLabels != null) {
                    topicLabel += ": " + topicLabels.get(k);
                }

                double[] distrs = topicWords[k].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + topicLabel
                        + ", " + topicWords[k].getCountSum()
                        + ", " + MiscUtils.formatDouble(eta[k])
                        + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
        }
    }
}
