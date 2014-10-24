package votepredictor;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import core.AbstractSampler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import optimization.RidgeLinearRegressionLBFGS;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class SLDAIdealPoint extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double rho;
    public double mu;
    public double sigma;
    public double epsilon; // learning rate when updating Xs and Ys
    public int numSteps = 20; // number of iterations when updating Xs and Ys

    // input
    protected int[][] words;
    protected ArrayList<Integer> docIndices;
    protected ArrayList<Integer> authorIndices;
    protected ArrayList<Integer> billIndices;
    protected int[] authors; // [D]: author of each document
    protected int[][] votes;
    protected boolean[][] validVotes;
    protected int K; // number of topics
    protected int D; // number of documents
    protected int A; // number of authors
    protected int B; // number of bills
    protected int V; // vocabulary size

    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;
    protected double[] eta; // regression parameters for topics
    protected double[] u; // [A]: authors' scores
    protected double[] x; // [B]
    protected double[] y; // [B]
    protected double[] authorMeans;

    // internal
    protected int numTokens;
    protected int numTokensChanged;
    protected int[][] authorDocIndices; // [A] x [D_a]: store the list of documents for each author
    protected int[] authorTokenCounts;  // [A]: store the total #tokens for each author
    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;

    public SLDAIdealPoint() {
        this.basename = "SLDA-ideal-point";
    }

    public SLDAIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public void setVoteVocab(ArrayList<String> voteVoc) {
        this.voteVocab = voteVoc;
    }

    public void configure(SLDAIdealPoint sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.rho,
                sampler.mu,
                sampler.sigma,
                sampler.epsilon,
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho,
            double mu, // mean of Gaussian for regression parameters
            double sigma, // stadard deviation of Gaussian for regression parameters
            double epsilon,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.hyperparams.add(alpha);
        this.hyperparams.add(beta);
        this.rho = rho;
        this.mu = mu;
        this.sigma = sigma;
        this.epsilon = epsilon;

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

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- num word types:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- epsilon:\t" + MiscUtils.formatDouble(epsilon));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_e-").append(formatter.format(epsilon))
                .append("_r-").append(formatter.format(rho))
                .append("_m-").append(formatter.format(mu))
                .append("_s-").append(formatter.format(sigma));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
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

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(this.getSamplerFolderPath())
                .append("\nCurrent thread: ").append(Thread.currentThread().getId());
        return str.toString();
    }

    protected void prepareDataStatistics() {
        // statistics
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += words[d].length;
        }

        // author document list
        ArrayList<Integer>[] authorDocList = new ArrayList[A];
        for (int a = 0; a < A; a++) {
            authorDocList[a] = new ArrayList<Integer>();
        }
        for (int d = 0; d < D; d++) {
            if (words[d].length > 0) {
                authorDocList[authors[d]].add(d);
            }
        }
        this.authorDocIndices = new int[A][];
        this.authorTokenCounts = new int[A];
        for (int a = 0; a < A; a++) {
            this.authorDocIndices[a] = new int[authorDocList[a].size()];
            for (int dd = 0; dd < this.authorDocIndices[a].length; dd++) {
                this.authorDocIndices[a][dd] = authorDocList[a].get(dd);
                this.authorTokenCounts[a] += words[authorDocIndices[a][dd]].length;
            }
        }
    }

    /**
     * Set training data.
     *
     * @param docIndices Indices of selected documents
     * @param words Document words
     * @param authors Document authors
     * @param votes All votes
     * @param authorIndices Indices of selected authors
     * @param billIndices Indices of selected bills
     * @param trainVotes Training votes
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
        // list of authors
        this.authorIndices = authorIndices;
        if (authorIndices == null) {
            this.authorIndices = new ArrayList<>();
            for (int aa = 0; aa < votes.length; aa++) {
                this.authorIndices.add(aa);
            }
        }
        this.A = this.authorIndices.size();

        HashMap<Integer, Integer> inverseAuthorMap = new HashMap<>();
        for (int ii = 0; ii < A; ii++) {
            int aa = this.authorIndices.get(ii);
            inverseAuthorMap.put(aa, ii);
        }

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
        this.validVotes = new boolean[A][B];
        for (int ii = 0; ii < A; ii++) {
            int aa = this.authorIndices.get(ii);
            for (int jj = 0; jj < B; jj++) {
                int bb = this.billIndices.get(jj);
                this.votes[ii][jj] = votes[aa][bb];
                this.validVotes[ii][jj] = trainVotes[aa][bb];
            }
        }

        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.authors = new int[D];
        for (int ii = 0; ii < this.D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = words[dd];
            this.authors[ii] = inverseAuthorMap.get(authors[dd]);
        }

        this.prepareDataStatistics();

        if (verbose) {
            logln("--- # documents:\t" + D);
            int numNonEmptyDocs = 0;
            for (int a = 0; a < A; a++) {
                numNonEmptyDocs += this.authorDocIndices[a].length;
            }
            logln("--- # non-empty documents:\t" + numNonEmptyDocs);
            int numNonEmptyAuthors = 0;
            for (int a = 0; a < A; a++) {
                if (authorDocIndices[a].length != 0) {
                    numNonEmptyAuthors++;
                }
            }
            logln("--- # speakers:\t" + A);
            logln("--- # non-empty speakers:\t" + numNonEmptyAuthors);
            logln("--- # tokens:\t" + numTokens
                    + ", " + StatUtils.sum(authorTokenCounts));
        }
    }

    /**
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @param testVotes Indicators of test votes
     * @return Predicted probabilities
     */
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

    /**
     * Make prediction for held-out voters using their text using the final
     * learned model. This can only make predictions on existing bills, so no
     * billIndices are needed.
     *
     * @param stateFile State file
     * @param docIndices List of selected document indices
     * @param words Documents
     * @param authors Voters
     * @param authorIndices List of test (held-out) voters
     * @param testVotes Indices of test votes
     * @param predictionFile
     * @return Predicted probabilities
     */
    public SparseVector[] test(File stateFile,
            ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            ArrayList<Integer> authorIndices,
            boolean[][] testVotes,
            File predictionFile) {
        if (authorIndices == null) {
            throw new RuntimeException("List of test authors is null");
        }

        if (stateFile == null) {
            stateFile = getFinalStateFile();
        }

        if (verbose) {
            logln("Setting up test ...");
            logln("--- state file: " + stateFile);
            logln("--- # test documents: " + docIndices.size());
            logln("--- # test authors: " + authorIndices.size());
        }

        // sample topic assignments for test documents
        this.sampleNewDocuments(getFinalStateFile(), null, words, docIndices);

        // predict author scores
        int testA = authorIndices.size();
        double[] predAuthorScores = new double[testA];
        double[] predAuthorDens = new double[testA];
        for (int ii = 0; ii < docIndices.size(); ii++) {
            int dd = docIndices.get(ii);
            int author = authors[dd];
            int aa = authorIndices.indexOf(author);
            if (aa < 0) {
                throw new RuntimeException("aa = " + aa + ". " + author);
            }
            for (int kk : docTopics[ii].getSparseCounts().getIndices()) {
                int count = docTopics[ii].getCount(kk);
                predAuthorScores[aa] += count * eta[kk];
                predAuthorDens[aa] += count;
            }
        }
        for (int aa = 0; aa < testA; aa++) {
            predAuthorScores[aa] /= predAuthorDens[aa];
        }

        // predict vote probabilities
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < testA; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector();
            double authorScore = predAuthorScores[aa];
            for (int bb = 0; bb < testVotes[author].length; bb++) {
                if (testVotes[author][bb]) {
                    double score = Math.exp(authorScore * x[bb] + y[bb]);
                    double val = score / (1 + score);
                    predictions[author].set(bb, val);
                }
            }
        }

        // output predictions
        if (predictionFile != null) {
            AbstractVotePredictor.outputPredictions(predictionFile, null, predictions);
        }

        return predictions;
    }

    /**
     * Sample topic assignments for all tokens in a set of test documents.
     *
     * @param stateFile
     * @param docTopicFile
     * @param testWords
     * @param testDocIndices
     */
    private void sampleNewDocuments(
            File stateFile,
            File docTopicFile,
            int[][] testWords,
            ArrayList<Integer> testDocIndices) {
        if (verbose) {
            System.out.println();
            logln("Perform regression using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
        }

        // input model
        inputModel(stateFile.getAbsolutePath());

        // set up test data
        this.docIndices = testDocIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < testWords.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.numTokens = 0;
        this.D = this.docIndices.size();
        this.words = new int[D][];
        for (int ii = 0; ii < D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = testWords[dd];
            this.numTokens += this.words[ii].length;
        }

        // set up model
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        // sample
        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;
            if (isReporting()) {
                String str = "Iter " + iter + "/" + MAX_ITER + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            long topicTime;
            if (iter == 0) {
                topicTime = sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);
            } else {
                topicTime = sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
            }

            if (isReporting()) {
                logln("--- --- Time. Topic: " + topicTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n\n");
            }
        }

        // output documents' distribution over topics (if necessary)
        if (docTopicFile != null) {
            outputDocTopics(docTopicFile);
        }
    }

    private void outputDocTopics(File file) {
        if (verbose) {
            logln("Outputing documents' topic distributions to " + file);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int ii = 0; ii < D; ii++) {
                int dd = this.docIndices.get(ii);
                writer.write(ii + "\t" + dd + "\t" + DirMult.output(docTopics[ii]) + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
    }

    private void inputDocTopics(File file) {
        if (verbose) {
            logln("Inputing documents' topic distributions from " + file);
        }
        try {
            this.docTopics = new DirMult[D];
            BufferedReader reader = IOUtils.getBufferedReader(file);
            String line;
            for (int ii = 0; ii < D; ii++) {
                line = reader.readLine();
                this.docTopics[ii] = DirMult.input(line.split("\t")[1]);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + file);
        }
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }
        initializeModelStructure(null);
        initializeDataStructure();
        initializeUXY();
        initializeAssignments();
        updateEtas();

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \n" + getCurrentState());
            getLogLikelihood();
        }
    }

    public void initialize(double[][] seededTopics) {
        if (verbose) {
            logln("Initializing ...");
        }
        initializeModelStructure(seededTopics);
        initializeDataStructure();
        initializeUXY();
        initializeAssignments();
        updateEtas();

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
                for (int kk = 0; kk < K; kk++) {
                    for (int vv = 0; vv < V; vv++) {
                        seededTopics[kk][vv] += (seededTopics[kk][vv] + 1.0 / V) / 2;
                    }
                }

                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, seededTopics[k]);
            } else {
                topicWords[k] = new DirMult(V, hyperparams.get(BETA) * V, 1.0 / V);
            }
        }

        eta = new double[K];
        for (int k = 0; k < K; k++) {
            eta[k] = SamplerUtils.getGaussian(mu, sigma);
        }

        x = new double[B];
        y = new double[B];
        u = new double[A];
    }

    protected void initializeDataStructure() {
        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
        }

        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        authorMeans = new double[A];
    }

    protected void initializeUXY() {
        if (verbose) {
            logln("--- Initializing UXY using Bayesian ideal point ...");
        }
        BayesianIdealPoint ip = new BayesianIdealPoint("bayesian-ideal-point");
        ip.configure(1.0, 0.01, 10000);

        File ipFolder = new File(folder, ip.getName());
        File ipFile = new File(ipFolder, this.basename + ".init");
        if (ipFile.exists()) {
            if (verbose) {
                logln("--- --- File exists. Loading from " + ipFile);
            }
            ip.input(ipFile);
        } else {
            if (verbose) {
                logln("--- --- File not exists. " + ipFile);
                logln("--- --- Running ideal point model ...");
            }
            ip.setTrain(votes, null, null, validVotes);
            ip.train();
            IOUtils.createFolder(ipFolder);
            ip.output(ipFile);
        }
        u = ip.getUs();
        x = ip.getXs();
        y = ip.getYs();
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    protected void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }

        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);

//        for (int d = 0; d < D; d++) {
//            for (int n = 0; n < words[d].length; n++) {
//                z[d][n] = rand.nextInt(K);
//                docTopics[d].increment(z[d][n]);
//                topicWords[z[d][n]].increment(words[d][n]);
//            }
//        }
    }

    @Override
    public void iterate() {
        if (verbose) {
            logln("Iterating ...");
        }
        logLikelihoods = new ArrayList<Double>();

        File reportFolderPath = new File(getSamplerFolderPath(), ReportFolder);
        try {
            if (report) {
                IOUtils.createFolder(reportFolderPath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while creating report folder."
                    + " " + reportFolderPath);
        }

        if (log && !isLogging()) {
            openLogger();
        }

        logln(getClass().toString());
        startTime = System.currentTimeMillis();

        for (iter = 0; iter < MAX_ITER; iter++) {
            numTokensChanged = 0;
            if (isReporting()) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                String str = "\n\nIter " + iter + "/" + MAX_ITER
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            long topicTime = sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            // L-BFGS to update etas
            long etaTime = updateEtas();

            // gradient ascent to update Xs and Ys
            long uxyTime = updateUXY();

            // parameter optimization
            if (iter % LAG == 0 && iter >= BURN_IN) {
                if (paramOptimized) { // slice sampling
                    sliceSample();
                    ArrayList<Double> sparams = new ArrayList<Double>();
                    for (double param : this.hyperparams) {
                        sparams.add(param);
                    }
                    this.sampledParams.add(sparams);

                    if (verbose) {
                        for (double p : sparams) {
                            System.out.println(p);
                        }
                    }
                }
            }

            if (isReporting()) {
                logln("--- --- Time. Topic: " + topicTime
                        + ". Eta: " + etaTime
                        + ". XY: " + uxyTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # token changed: " + numTokensChanged
                        + ". change ratio: " + (double) numTokensChanged / numTokens
                        + "\n\n");
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
                outputTopicTopWords(new File(reportFolderPath,
                        "iter-" + iter + "-" + TopWordFile), 15);
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, "iter-" + iter + ".zip"));
            outputTopicTopWords(new File(reportFolderPath,
                    "iter-" + iter + "-" + TopWordFile), 15);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
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
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    topicWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][n]);
                    if (observe) {
                        authorMeans[authors[d]] -= eta[z[d][n]] / authorTokenCounts[authors[d]];
                    }
                }

                double[] logprobs = new double[K];
                for (int k = 0; k < K; k++) {
                    logprobs[k]
                            = Math.log(docTopics[d].getCount(k) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[k].getProbability(words[d][n]));
                    if (observe) {
                        double aMean = authorMeans[authors[d]] + eta[k] / authorTokenCounts[authors[d]];
                        double resLLh = StatUtils.logNormalProbability(u[authors[d]], aMean, Math.sqrt(rho));
                        logprobs[k] += resLLh;
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
                    if (observe) {
                        authorMeans[authors[d]] += eta[z[d][n]] / authorTokenCounts[authors[d]];
                    }
                }
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Optimize etas using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        long sTime = System.currentTimeMillis();
        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(K);
            for (int dd : authorDocIndices[aa]) {
                for (int kk : docTopics[dd].getSparseCounts().getIndices()) {
                    double val = (double) docTopics[dd].getSparseCounts().getCount(kk)
                            / authorTokenCounts[aa];
                    designMatrix[aa].change(kk, val);
                }
            }
        }

        RidgeLinearRegressionLBFGS optimizable = new RidgeLinearRegressionLBFGS(
                u, eta, designMatrix, rho, mu, sigma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (isReporting()) {
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
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Update ideal point model's parameters using gradient ascent.
     *
     * @return Elapsed time
     */
    private long updateUXY() {
        long sTime = System.currentTimeMillis();
        for (int ii = 0; ii < numSteps; ii++) {
            updateUs();
            updateXYs();
        }
        return System.currentTimeMillis() - sTime;
    }

    private void updateUs() {
        for (int a = 0; a < A; a++) {
            double grad = 0.0;
            for (int b = 0; b < votes[a].length; b++) { // likelihood
                if (validVotes[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    grad += x[b] * (votes[a][b] - prob); // only work for 0 and 1
                }
            }
            grad -= (u[a] - authorMeans[a]) / sigma; // prior
            u[a] += epsilon * grad; // update
        }
    }

    public void updateXYs() {
        for (int b = 0; b < B; b++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int a = 0; a < A; a++) {
                if (validVotes[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    gradX += u[a] * (votes[a][b] - score / (1 + score));
                    gradY += votes[a][b] - score / (1 + score);
                }
            }
            // prior
            gradX -= (x[b] - mu) / sigma;
            gradY -= (y[b] - mu) / sigma;

            // update
            x[b] += epsilon * gradX;
            y[b] += epsilon * gradY;
        }
    }

    private double getVoteLogLikelihood() {
        double voteLlh = 0.0;
        for (int a = 0; a < A; a++) {
            voteLlh += computeAuthorVoteLogLikelihood(a, u[a]);
        }
        return voteLlh;
    }

    private double computeAuthorVoteLogLikelihood(int author, double authorVal) {
        double llh = 0.0;
        for (int b = 0; b < B; b++) {
            if (validVotes[author][b]) {
                double score = authorVal * x[b] + y[b];
                llh += votes[author][b] * score - Math.log(1 + Math.exp(score));
            }
        }
        return llh;
    }

    @Override
    public double getLogLikelihood() {
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood();
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood();
        }

        double voteLlh = getVoteLogLikelihood();
        double llh = wordLlh + topicLlh + voteLlh;
        if (isReporting()) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". vote: " + MiscUtils.formatDouble(voteLlh)
                    + ". llh = " + MiscUtils.formatDouble(llh));
        }

        return llh;
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
        for (int d = 0; d < D; d++) {
            docTopics[d].validate(msg);
        }
        for (int k = 0; k < K; k++) {
            topicWords[k].validate(msg);
        }
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
            // train author scores
            modelStr.append(A).append("\n");
            for (int aa = 0; aa < A; aa++) {
                modelStr.append(u[aa]).append("\n");
            }
            // train author scores
            modelStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                modelStr.append(x[bb]).append("\n");
                modelStr.append(y[bb]).append("\n");
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
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
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

    protected void inputModel(String zipFilepath) {
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

            this.A = Integer.parseInt(reader.readLine());
            u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                u[aa] = Double.parseDouble(reader.readLine());
            }

            this.B = Integer.parseInt(reader.readLine());
            x = new double[B];
            y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                x[bb] = Double.parseDouble(reader.readLine());
                y[bb] = Double.parseDouble(reader.readLine());
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    protected void inputAssignments(String zipFilepath) throws Exception {
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
                    topicLabel = topicLabels.get(k);
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
    
    public void getAuthorTopicProportions() {
        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(K);
            for (int dd : authorDocIndices[aa]) {
                for (int kk : docTopics[dd].getSparseCounts().getIndices()) {
                    double val = (double) docTopics[dd].getSparseCounts().getCount(kk)
                            / authorTokenCounts[aa];
                    designMatrix[aa].change(kk, val);
                }
            }
        }
    }

    /**
     * Objective function to optimize for eta while keeping other fixed.
     */
    class Objective implements Optimizable.ByGradientValue {

        double[] params; // etas
        double[][] authorZs;
        double[] voteXs;
        double[] voteYs;

        public Objective(double[] curparams,
                double[][] az, double[] vx, double[] vy) {
            this.authorZs = az;
            this.voteXs = vx;
            this.voteYs = vy;
            this.params = new double[curparams.length];
            System.arraycopy(curparams, 0, this.params, 0, params.length);
        }

        @Override
        public double getValue() {
            double val = 0.0;
            for (int a = 0; a < A; a++) {
                double dotprod = 0.0;
                for (int k = 0; k < K; k++) {
                    dotprod += params[k] * authorZs[a][k];
                }

                for (int b = 0; b < B; b++) {
                    if (validVotes[a][b]) {
                        double score = voteXs[b] * dotprod + voteYs[b];
                        val += votes[a][b] * score - Math.log(1 + Math.exp(score));
                    }
                }
            }
            return val;
        }

        @Override
        public void getValueGradient(double[] gradient) {
            Arrays.fill(gradient, 0.0);
            for (int a = 0; a < A; a++) {
                double dotprod = 0.0;
                for (int k = 0; k < K; k++) {
                    dotprod += params[k] * authorZs[a][k];
                }
                for (int b = 0; b < B; b++) {
                    if (validVotes[a][b]) {
                        double score = Math.exp(voteXs[b] * dotprod + voteYs[b]);
                        for (int k = 0; k < K; k++) {
                            gradient[k] += voteXs[b] * authorZs[a][k]
                                    * (votes[a][b] - score / (1 + score));
                        }
                    }
                }
            }
        }

        @Override
        public int getNumParameters() {
            return this.params.length;
        }

        @Override
        public double getParameter(int i) {
            return params[i];
        }

        @Override
        public void getParameters(double[] buffer) {
            assert (buffer.length == params.length);
            System.arraycopy(params, 0, buffer, 0, buffer.length);
        }

        @Override
        public void setParameter(int i, double r) {
            this.params[i] = r;
        }

        @Override
        public void setParameters(double[] newParameters) {
            assert (newParameters.length == params.length);
            System.arraycopy(newParameters, 0, params, 0, params.length);
        }
    }

    /**
     * Run Gibbs sampling on test data using multiple models learned which are
     * stored in the ReportFolder. The runs on multiple models are parallel.
     *
     * @param newWords Words of new documents
     * @param newDocIndices Indices of selected documents
     * @param newAuthors
     * @param newAuthorIndices
     * @param testVotes
     * @param iterPredFolder Output folder
     * @param sampler The configured sampler
     * @return Predicted probabilities
     */
    public static SparseVector[] parallelTest(ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File iterPredFolder,
            SLDAIdealPoint sampler) {
        File reportFolder = new File(sampler.getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder not found. " + reportFolder);
        }
        SparseVector[] predictions = null;
        String[] filenames = reportFolder.list();
        try {
            IOUtils.createFolder(iterPredFolder);
            ArrayList<Thread> threads = new ArrayList<Thread>();
            ArrayList<File> partPredFiles = new ArrayList<>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }

                File stateFile = new File(reportFolder, filename);
                File partialResultFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".txt");
                SamplingTestDocumentsRunner runner = new SamplingTestDocumentsRunner(
                        sampler, stateFile, newDocIndices, newWords,
                        newAuthors, newAuthorIndices, testVotes, partialResultFile);
                Thread thread = new Thread(runner);
                threads.add(thread);
                partPredFiles.add(partialResultFile);
            }

            // run MAX_NUM_PARALLEL_THREADS threads at a time
            runThreads(threads);

            // average over multiple predictions
            for (File partPredFile : partPredFiles) {
                SparseVector[] partPredictions = AbstractVotePredictor.inputPredictions(partPredFile);
                if (predictions == null) {
                    predictions = partPredictions;
                } else {
                    if (predictions.length != partPredictions.length) {
                        throw new RuntimeException("Mismatch");
                    }
                    for (int aa : newAuthorIndices) {
                        predictions[aa].add(partPredictions[aa]);
                    }
                }
            }
            for (int aa : newAuthorIndices) {
                predictions[aa].scale(1.0 / partPredFiles.size());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during parallel test.");
        }
        return predictions;
    }
}

class SamplingTestDocumentsRunner implements Runnable {

    SLDAIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public SamplingTestDocumentsRunner(SLDAIdealPoint sampler,
            File stateFile,
            ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File outputFile) {
        this.sampler = sampler;
        this.stateFile = stateFile;
        this.testDocIndices = newDocIndices;
        this.testWords = newWords;
        this.testAuthors = newAuthors;
        this.testAuthorIndices = newAuthorIndices;
        this.testVotes = testVotes;
        this.predictionFile = outputFile;
    }

    @Override
    public void run() {
        SLDAIdealPoint testSampler = new SLDAIdealPoint();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.test(stateFile, testDocIndices, testWords, testAuthors,
                    testAuthorIndices, testVotes, predictionFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
