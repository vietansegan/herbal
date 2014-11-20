package votepredictor;

import core.AbstractSampler;
import edu.stanford.nlp.optimization.DiffFunction;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import optimization.OWLQN;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;

/**
 *
 * @author vietan
 */
public class SLDAMultIdealPoint extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double rho;
    public double mu;       // topic regression parameter mean
    public double sigma;    // topic regression parameter variance
    protected double ipSigma; // bill ideal point variance
    protected double l1; // lexical regression l1-regularizer
    protected double l2; // lexical regression l2-regularizer
    // input
    protected int[][] words;
    protected ArrayList<Integer> docIndices;    // potentially not needed
    protected ArrayList<Integer> authorIndices; // potentially not needed
    protected ArrayList<Integer> billIndices;   // potentially not needed
    protected int[] authors; // [D]: author of each document
    protected int[][] votes;
    protected boolean[][] validVotes;
    protected double[][] issuePhis;
    protected int V; // vocabulary size
    protected int K; // number of issues
    // derive
    protected int D; // number of documents
    protected int A; // number of authors
    protected int B; // number of bills
    protected int L; // number of levels
    protected boolean[] validAs; // flag voters with no training vote
    protected boolean[] validBs; // flag bills with no training vote
    protected SparseVector[] authorLexDsgMatrix; // for lexical regression
    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;
    protected double[] eta; // regression parameters for topics
    protected SparseVector[] xy; // [B][K + 1]
    protected SparseVector[] us; // voters' multi-dimensional ideal points
    protected SparseVector[] za;

    // internal
    protected int numTokens;
    protected int numTokensChanged;
    protected int[][] authorDocIndices; // [A] x [D_a]: store the list of documents for each author
//    protected int[] authorTokenCounts;  // [A]: store the total #tokens for each author
    protected double[] authorInversedTokenCounts;
    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;

    public SLDAMultIdealPoint() {
        this.basename = "SLDA-mult-ideal-point";
    }

    public SLDAMultIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public void setVoteVocab(ArrayList<String> voteVoc) {
        this.voteVocab = voteVoc;
    }

    public void configure(
            String folder,
            int V, int K,
            double alpha,
            double beta,
            double rho,
            double mu, // mean of Gaussian for regression parameters
            double sigma, // stadard deviation of Gaussian for regression parameters
            double ipSigma,
            double l1,
            double l2,
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
        this.ipSigma = ipSigma;
        this.l1 = l1;
        this.l2 = l2;

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
            logln("--- ideal point sigma:\t" + MiscUtils.formatDouble(ipSigma));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2));
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
                .append("_r-").append(formatter.format(rho))
                .append("_m-").append(formatter.format(mu))
                .append("_s-").append(formatter.format(sigma))
                .append("_is-").append(formatter.format(ipSigma))
                .append("_l1-").append(formatter.format(l1))
                .append("_l2-").append(formatter.format(l2))
                .append("_m-").append(formatter.format(mu))
                .append("_s-").append(formatter.format(sigma));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(this.getSamplerFolderPath())
                .append("\nCurrent thread: ").append(Thread.currentThread().getId());
        return str.toString();
    }

    private int getVote(int aa, int bb) {
        return this.votes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    private boolean isValidVote(int aa, int bb) {
        return this.validVotes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
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
        int[] authorTokenCounts = new int[A];
        for (int a = 0; a < A; a++) {
            this.authorDocIndices[a] = new int[authorDocList[a].size()];
            for (int dd = 0; dd < this.authorDocIndices[a].length; dd++) {
                this.authorDocIndices[a][dd] = authorDocList[a].get(dd);
                authorTokenCounts[a] += words[authorDocIndices[a][dd]].length;
            }
        }
        this.authorInversedTokenCounts = new double[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorInversedTokenCounts[aa] = 1.0 / authorTokenCounts[aa];
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
     * @param maskedVotes Valid votes
     */
    public void setupData(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] maskedVotes) {
        // list of authors
        this.authorIndices = authorIndices;
        if (authorIndices == null) {
            this.authorIndices = new ArrayList<>();
            for (int aa = 0; aa < maskedVotes.length; aa++) {
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
            for (int bb = 0; bb < maskedVotes[0].length; bb++) {
                this.billIndices.add(bb);
            }
        }
        this.B = this.billIndices.size();

        // documents
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

        this.validVotes = maskedVotes;
        this.votes = votes; // null if test

        // skip voters/bills which don't have any vote
        this.validAs = new boolean[A];
        this.validBs = new boolean[B];
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (validVotes[this.authorIndices.get(aa)][this.billIndices.get(bb)]) {
                    this.validAs[aa] = true;
                    this.validBs[bb] = true;
                }
            }
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
            logln("--- # tokens:\t" + numTokens);
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
                    double dotprod = xy[bb].get(K); // y
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa].get(kk) * xy[bb].get(kk);
                    }
                    double score = Math.exp(dotprod);
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
     * @param partAuthorScoreFile
     * @param partVoteScoreFile
     * @param assignmentFile
     * @return Predicted probabilities
     */
    public SparseVector[] test(File stateFile,
            ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            ArrayList<Integer> authorIndices,
            boolean[][] testVotes,
            File predictionFile,
            File partAuthorScoreFile,
            File partVoteScoreFile,
            File assignmentFile) {
        if (authorIndices == null) {
            throw new RuntimeException("List of test authors is null");
        }

        if (stateFile == null) {
            stateFile = getFinalStateFile();
        }

        if (verbose) {
            logln("Setting up test ...");
            logln("--- state file: " + stateFile);
        }

        setTestConfigurations(100, 250, 10, 5);

        // set up test data
        setupData(docIndices, words, authors, null, authorIndices, null, testVotes);

        // sample assignments for new documents
        sampleNewDocuments(getFinalStateFile(), assignmentFile);

        SparseVector[] predictions = makePredictions();

        if (predictionFile != null) { // output predictions
            AbstractVotePredictor.outputPredictions(predictionFile, null, predictions);
        }

        return predictions;
    }

    /**
     * Sample topic assignments for all tokens in a set of test documents.
     *
     * @param stateFile
     * @param testWords
     * @param testDocIndices
     * @param assignmentFile
     */
    private void sampleNewDocuments(
            File stateFile,
            File assignmentFile) {
        if (verbose) {
            System.out.println();
            logln("Perform regression using model from " + stateFile);
            logln("--- Test burn-in: " + this.testBurnIn);
            logln("--- Test max-iter: " + this.testMaxIter);
            logln("--- Test sample-lag: " + this.testSampleLag);
            logln("--- Test report-interval: " + this.testRepInterval);
        }

        // input model
        inputModel(stateFile.getAbsolutePath());
        inputBillScore(stateFile.getAbsolutePath());
        initializeDataStructure();

        // sample
        for (iter = 0; iter < testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter + "\n" + getCurrentState();
                if (iter < testBurnIn) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED, isReporting);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED, isReporting);
            }

            if (debug) {
                validate("test iter " + iter);
            }
        }

        if (assignmentFile != null) {
            StringBuilder assignStr = new StringBuilder();
            for (int d = 0; d < D; d++) {
                assignStr.append(d).append("\n");
                assignStr.append(DirMult.output(docTopics[d])).append("\n");
                for (int n = 0; n < z[d].length; n++) {
                    assignStr.append(z[d][n]).append("\t");
                }
                assignStr.append("\n");
            }

            try { // output to a compressed file
                ArrayList<String> contentStrs = new ArrayList<>();
                contentStrs.add(assignStr.toString());

                String filename = IOUtils.removeExtension(IOUtils.getFilename(assignmentFile.getAbsolutePath()));
                ArrayList<String> entryFiles = new ArrayList<>();
                entryFiles.add(filename + AssignmentFileExt);

                this.outputZipFile(assignmentFile.getAbsolutePath(), contentStrs, entryFiles);
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException("Exception while outputing to " + assignmentFile);
            }
        }
    }

    /**
     * Make predictions on test data.
     *
     * @return Predictions
     */
    private SparseVector[] makePredictions() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = this.authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < validVotes[author].length; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk : this.za[aa].getIndices()) {
                        dotprod += za[aa].get(kk) * eta[kk] * xy[bb].get(kk);
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (score + 1);
                    predictions[author].set(bb, prob);
                }
            }
        }
        return predictions;
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;
        initializeModelStructure(null);
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

        this.xy = new SparseVector[B];
        for (int bb = 0; bb < B; bb++) {
            if (validBs[bb]) {
                this.xy[bb] = new SparseVector(K + 1);
                for (int kk = 0; kk < K + 1; kk++) {
                    this.xy[bb].set(kk, SamplerUtils.getGaussian(mu, ipSigma));
                }
            }
        }
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

        this.us = new SparseVector[A];
        this.za = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.za[aa] = new SparseVector(K);
            this.us[aa] = new SparseVector(K);
            for (int kk = 0; kk < K; kk++) {
                this.us[aa].set(kk, SamplerUtils.getGaussian(mu, ipSigma));
            }
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

        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED, true);
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
            isReporting = isReporting();
            if (isReporting) {
                double loglikelihood = this.getLogLikelihood();
                logLikelihoods.add(loglikelihood);
                System.out.println();
                String str = "\nIter " + iter + "/" + MAX_ITER
                        + "\t llh = " + MiscUtils.formatDouble(loglikelihood)
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            updateEtas();
            updateUs();
            updateXY();
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED, isReporting);

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
     * @param isRep
     * @return Elapsed time
     */
    protected long sampleZs(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe, boolean isRep) {
        if (isRep) {
            logln("+++ Sampling assignments ...");
        }
        numTokensChanged = 0;
        long sTime = System.currentTimeMillis();
        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int nn = 0; nn < words[d].length; nn++) {
                if (removeFromModel) {
                    topicWords[z[d][nn]].decrement(words[d][nn]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][nn]);
                    za[aa].change(z[d][nn], -authorInversedTokenCounts[aa]);
                }

                double[] logprobs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    logprobs[kk]
                            = Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[kk].getProbability(words[d][nn]));
                    if (observe) {
                        double mean = (za[aa].get(kk) + authorInversedTokenCounts[aa]) * eta[kk];
                        double voteLlh = StatUtils.logNormalProbability(us[aa].get(kk), mean, Math.sqrt(rho));
                        logprobs[kk] += voteLlh;
                    }
                }
                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
                if (sampledZ == K) {
                    logln("iter = " + iter + ". d = " + d + ". n = " + nn);
                    for (int kk = 0; kk < K; kk++) {
                        logln("k = " + kk
                                + ". " + (Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA)))
                                + ". " + (Math.log(topicWords[kk].getProbability(words[d][nn]))));
                    }
                    throw new RuntimeException("Out-of-bound sample. "
                            + "SampledZ = " + sampledZ);
                }

                if (z[d][nn] != sampledZ) {
                    numTokensChanged++; // for debugging
                }
                // update
                z[d][nn] = sampledZ;

                if (addToModel) {
                    topicWords[z[d][nn]].increment(words[d][nn]);
                }
                if (addToData) {
                    docTopics[d].increment(z[d][nn]);
                    za[aa].change(z[d][nn], authorInversedTokenCounts[aa]);
                }
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isRep) {
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

        OWLQN minimizer = new OWLQN();
        minimizer.setQuiet(true);
        minimizer.setMaxIters(100);
        EtaDiffFunc diff = new EtaDiffFunc();
        double[] tempEta = new double[K];
        System.arraycopy(eta, 0, tempEta, 0, K);
        minimizer.minimize(diff, tempEta, 0.0);
        System.arraycopy(tempEta, 0, eta, 0, K);

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    protected long updateUs() {
        if (isReporting) {
            logln("+++ Updating Us ...");
        }
        long sTime = System.currentTimeMillis();
        for (int aa = 0; aa < A; aa++) {
            if (!this.validAs[aa]) {
                continue;
            }
            OWLQN minimizer = new OWLQN();
            minimizer.setQuiet(true);
            minimizer.setMaxIters(100);
            UDiffFunc udiff = new UDiffFunc(aa);
            double[] tempU = new double[K];
            for (int kk : us[aa].getIndices()) {
                tempU[kk] = us[aa].get(kk);
            }
            minimizer.minimize(udiff, tempU, l1);
            us[aa] = new SparseVector(tempU);
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    /**
     * Update U, X and Y.
     *
     * @return Elapsed time
     */
    private long updateXY() {
        if (isReporting) {
            logln("+++ Updating XYs ...");
        }
        long sTime = System.currentTimeMillis();

        for (int bb = 0; bb < B; bb++) {
            if (!this.validBs[bb]) {
                continue;
            }
            OWLQN minimizer = new OWLQN();
            minimizer.setQuiet(true);
            minimizer.setMaxIters(100);
            XYDiffFunc xydiff = new XYDiffFunc(bb);
            double[] tempXY = new double[K + 1];
            for (int kk : xy[bb].getIndices()) {
                tempXY[kk] = xy[bb].get(kk);
            }
            minimizer.minimize(xydiff, tempXY, 0.0);
            xy[bb] = new SparseVector(tempXY);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
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

        double voteLlh = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa].get(kk) * xy[bb].get(kk);
                    }
                    voteLlh += getVote(aa, bb) * dotprod - Math.log(1 + Math.exp(dotprod));
                }
            }
        }
        double llh = voteLlh + wordLlh + topicLlh;
        if (isReporting) {
            logln("--- --- word-llh: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic-llh: " + MiscUtils.formatDouble(topicLlh)
                    + ". vote-llh: " + MiscUtils.formatDouble(topicLlh)
                    + ". total-llh: " + MiscUtils.formatDouble(llh));
        }
        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        throw new RuntimeException("Currently not supported");
    }

    @Override
    public void validate(String msg) {
        logln("Validating ... " + msg + ". Turn this off by disable \"d\".");
        for (int kk = 0; kk < K; kk++) {
            this.topicWords[kk].validate(msg);
        }
        int totalTokens = 0;
        for (int dd = 0; dd < D; dd++) {
            this.docTopics[dd].validate(msg);
            totalTokens += this.docTopics[dd].getCountSum();
        }
        if (totalTokens != numTokens) {
            throw new MismatchRuntimeException(totalTokens, numTokens);
        }
        for (int aa = 0; aa < A; aa++) {
            if (this.za[aa].isEmpty()) {
                continue;
            }
            double sum = this.za[aa].sum();
            if (Math.abs(sum - 1.0) > 0.00001) {
                throw new MismatchRuntimeException(Double.toString(sum), "1.0");
            }
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        // bills
        StringBuilder billStr = new StringBuilder();
        for (int bb = 0; bb < B; bb++) {
            billStr.append(bb).append("\t")
                    .append(SparseVector.output(xy[bb])).append("\n");
        }

        // model string
        StringBuilder modelStr = new StringBuilder();
        for (int kk = 0; kk < K; kk++) {
            modelStr.append(kk).append("\n");
            modelStr.append(eta[kk]).append("\n");
            modelStr.append(DirMult.output(topicWords[kk])).append("\n");
        }

        // assignment string
        StringBuilder assignStr = new StringBuilder();
        for (int d = 0; d < D; d++) {
            assignStr.append(d).append("\n");
            assignStr.append(DirMult.output(docTopics[d])).append("\n");
            for (int n = 0; n < z[d].length; n++) {
                assignStr.append(z[d][n]).append("\t");
            }
            assignStr.append("\n");
        }

        try { // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());
            contentStrs.add(billStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + ".bill");

            this.outputZipFile(filepath, contentStrs, entryFiles);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + filepath);
        }
    }

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }
        try {
            inputModel(filepath);
            inputBillScore(filepath);
            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + filepath);
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
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
                    + zipFilepath);
        }
    }

    public void inputBillScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading bill scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ".bill");
            xy = new SparseVector[B];
            for (int bb = 0; bb < B; bb++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != bb) {
                    throw new MismatchRuntimeException(Integer.parseInt(sline[0]), bb);
                }
                xy[bb] = SparseVector.input(sline[1]);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    public void outputTopicTopWords(File file, int numTopWords) {
        this.outputTopicTopWords(file, numTopWords, null);
    }

    public void outputAuthorIdealPoint(File file, ArrayList<String> authorNames) {
        if (verbose) {
            logln("Outputing author ideal points " + file);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int aa = 0; aa < A; aa++) {
                String authorName = Integer.toString(aa);
                if (authorNames != null) {
                    authorName = authorNames.get(aa);
                }
                writer.write(authorName);
                for (int kk = 0; kk < K; kk++) {
                    double idealpoint = za[aa].get(kk) * eta[kk];
                    writer.write("\t" + idealpoint);
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + file);
        }
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

    public SparseVector[] getAuthorFeatures() {
        SparseVector[] authorFeatures = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            authorFeatures[aa] = new SparseVector(2 * K);
            for (int kk = 0; kk < K; kk++) {
                authorFeatures[aa].set(kk, za[aa].get(kk));
            }
            for (int kk = 0; kk < K; kk++) {
                authorFeatures[aa].set(K + kk, za[aa].get(kk) * eta[kk]);
            }
        }
        return authorFeatures;
    }

    class EtaDiffFunc implements DiffFunction {

        @Override
        public int domainDimension() {
            return K;
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    double diff = us[aa].get(kk) - w[kk] * za[aa].get(kk);
                    llh += 0.5 * diff * diff / rho;
                }
            }

            double prior = 0.0;
            for (int kk = 0; kk < K; kk++) {
                prior += 0.5 * w[kk] * w[kk] / sigma;
            }
            return llh + prior;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[domainDimension()];
            for (int aa = 0; aa < A; aa++) {
                for (int kk : za[aa].getIndices()) {
                    grads[kk] += (za[aa].get(kk) * w[kk] - us[aa].get(kk))
                            * za[aa].get(kk) / rho;
                }
            }
            for (int kk = 0; kk < w.length; kk++) {
                grads[kk] += w[kk] / sigma;
            }
            return grads;
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
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += xy[bb].get(kk) * w[kk];
                    }
                    llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }
            double reg = 0.0;
            for (int kk = 0; kk < w.length; kk++) {
                double diff = w[kk] - za[aa].get(kk) * eta[kk];
                reg += 0.5 * diff * diff / ipSigma;
            }
            double val = -llh + reg;
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += xy[bb].get(kk) * w[kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] -= xy[bb].get(kk) * (getVote(aa, bb) - prob);
                    }
                }
            }
            for (int kk = 0; kk < w.length; kk++) {
                double diff = w[kk] - za[aa].get(kk) * eta[kk];
                grads[kk] += diff / ipSigma;
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
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = w[K];
                    for (int kk : xy[bb].getIndices()) {
                        dotprod += us[aa].get(kk) * w[kk];
                    }
                    llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }

            double reg = 0.0;
            for (int ii = 0; ii < w.length; ii++) {
                reg += 0.5 * w[ii] * w[ii] / ipSigma;
            }

            double val = -llh + reg;
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K + 1];
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = w[K];
                    for (int kk : xy[bb].getIndices()) {
                        dotprod += us[aa].get(kk) * w[kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk : xy[bb].getIndices()) {
                        grads[kk] -= us[aa].get(kk) * (getVote(aa, bb) - prob);
                    }
                    grads[K] -= 1.0 * (getVote(aa, bb) - prob);
                }
            }
            for (int kk = 0; kk < w.length; kk++) {
                grads[kk] += w[kk] / ipSigma;
            }
            return grads;
        }
    }
}
