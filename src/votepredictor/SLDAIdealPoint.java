package votepredictor;

import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import core.AbstractSampler;
import data.Vote;
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
import util.evaluation.Measurement;

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
    public double rate_alpha;
    public double rate_eta;
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
    protected int posAnchor;
    protected int negAnchor;

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
                sampler.rate_alpha,
                sampler.rate_eta,
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
            double rate_alpha,
            double rate_eta,
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
        this.rate_alpha = rate_alpha;
        this.rate_eta = rate_eta;

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
            logln("--- rate-alpha:\t" + MiscUtils.formatDouble(rate_alpha));
            logln("--- rate-eta:\t" + MiscUtils.formatDouble(rate_eta));
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
                .append("_ra-").append(formatter.format(rate_alpha))
                .append("_re-").append(formatter.format(rate_eta))
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

    public double[] getPredictedUs() {
        return this.authorMeans;
    }

    private int getVote(int aa, int bb) {
        return this.votes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    private boolean isValidVote(int aa, int bb) {
        return this.validVotes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
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

        this.votes = votes;
        this.validVotes = trainVotes;

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
    public SparseVector[] predict(boolean[][] testVotes) {
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(testVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (testVotes[author][bill]) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    /**
     * Evaluate prediction during training.
     */
    private void evaluate() {
        SparseVector[] predictions = predict(validVotes);
        ArrayList<Measurement> measurements = AbstractVotePredictor
                .evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> >>> " + m.getName() + ": " + m.getValue());
        }
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

        this.setTestConfigurations(50, 100, 10, 5);

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
                continue;
            }
            for (int kk : docTopics[ii].getSparseCounts().getIndices()) {
                int count = docTopics[ii].getCount(kk);
                predAuthorScores[aa] += count * eta[kk];
                predAuthorDens[aa] += count;
            }
        }

        this.authorMeans = new double[testA];
        for (int aa = 0; aa < testA; aa++) {
            this.authorMeans[aa] = predAuthorScores[aa] / predAuthorDens[aa];
        }

        // predict vote probabilities
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < testA; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(testVotes[author].length);
            double authorScore = this.authorMeans[aa];
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
        for (iter = 0; iter < testMaxIter; iter++) {
            boolean disp = verbose && iter % testRepInterval == 0;
            if (disp) {
                String str = "Iter " + iter + "/" + testMaxIter + "\n" + getCurrentState();
                if (iter < testBurnIn) {
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

            if (disp) {
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

    protected void outputDocTopics(File file) {
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

    protected void inputDocTopics(File file) {
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
            logln("--- Initializing random UXY using anchored legislators ...");
        }
        double anchorMean = 3.0;
        double anchorVar = 0.01;
        ArrayList<RankingItem<Integer>> rankAuthors = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            int withCount = 0;
            int againstCount = 0;
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    if (getVote(aa, bb) == Vote.WITH) {
                        withCount++;
                    } else if (getVote(aa, bb) == Vote.AGAINST) {
                        againstCount++;
                    }
                }
            }
            double val = (double) withCount / (againstCount + withCount);
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
                this.u[aa] = SamplerUtils.getGaussian(mu, sigma);
            }
        }

        this.x = new double[B];
        this.y = new double[B];
        for (int b = 0; b < B; b++) {
            this.x[b] = SamplerUtils.getGaussian(mu, sigma);
            this.y[b] = SamplerUtils.getGaussian(mu, sigma);
        }
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
            isReporting = isLogging();
            if (isReporting) {
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
                logln("--- --- positive anchor (" + posAnchor + "): "
                        + MiscUtils.formatDouble(u[posAnchor])
                        + ". negative anchor (" + negAnchor + "): "
                        + MiscUtils.formatDouble(u[negAnchor]));
                evaluate();
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

            if (isReporting) {
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
        numTokensChanged = 0;
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
        double aRate = getLearningRate();
        for (int aa = 0; aa < A; aa++) {
            double grad = 0.0;
            for (int bb = 0; bb < B; bb++) { // likelihood
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    double prob = score / (1 + score);
                    grad += x[bb] * (getVote(aa, bb) - prob); // only work for 0 and 1
                }
            }
            grad -= (u[aa] - authorMeans[aa]) / sigma; // prior
            u[aa] += aRate * grad; // update
        }
    }

    public void updateXYs() {
        double bRate = getLearningRate();
        for (int bb = 0; bb < B; bb++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    gradX += u[aa] * (getVote(aa, bb) - score / (1 + score));
                    gradY += getVote(aa, bb) - score / (1 + score);
                }
            }
            // prior
            gradX -= (x[bb] - mu) / sigma;
            gradY -= (y[bb] - mu) / sigma;

            // update
            x[bb] += bRate * gradX;
            y[bb] += bRate * gradY;
        }
    }

    public double getLearningRate() {
        return rate_eta * Math.pow(rate_alpha, -(double) iter / MAX_ITER);
    }

    private double getVoteLogLikelihood() {
        double voteLlh = 0.0;
        for (int a = 0; a < A; a++) {
            voteLlh += computeAuthorVoteLogLikelihood(a, u[a]);
        }
        return voteLlh;
    }

    private double computeAuthorVoteLogLikelihood(int aa, double authorVal) {
        double llh = 0.0;
        for (int bb = 0; bb < B; bb++) {
            if (isValidVote(aa, bb)) {
                double score = authorVal * x[bb] + y[bb];
                llh += getVote(aa, bb) * score - Math.log(1 + Math.exp(score));
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

    public double[][] getAuthorTopicProportions() {
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
        double[][] authorTopicProps = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa].normalize();
            authorTopicProps[aa] = designMatrix[aa].dense();
        }
        return authorTopicProps;
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
                    if (isValidVote(a, b)) {
                        double score = voteXs[b] * dotprod + voteYs[b];
                        val += getVote(a, b) * score - Math.log(1 + Math.exp(score));
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
                    if (isValidVote(a, b)) {
                        double score = Math.exp(voteXs[b] * dotprod + voteYs[b]);
                        for (int k = 0; k < K; k++) {
                            gradient[k] += voteXs[b] * authorZs[a][k]
                                    * (getVote(a, b) - score / (1 + score));
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
                SLDATestRunner runner = new SLDATestRunner(
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

class SLDATestRunner implements Runnable {

    SLDAIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public SLDATestRunner(SLDAIdealPoint sampler,
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
