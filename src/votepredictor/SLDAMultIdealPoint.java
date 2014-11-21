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
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.evaluation.Measurement;

/**
 *
 * @author vietan
 */
public class SLDAMultIdealPoint extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double etaL2; // l2-regularizer for eta's
    protected double l1; // l1-regularizer for x and y
    protected double l2; // l2-regularizer for x and y
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
    // configure
    protected boolean mh;
    // derive
    protected int D; // number of documents
    protected int A; // number of authors
    protected int B; // number of bills
    protected boolean[] validAs; // flag voters with no training vote
    protected boolean[] validBs; // flag bills with no training vote
    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;
    protected double[] eta; // regression parameters for topics
    protected SparseVector[] xy; // [B][K + 1]
    // internal
    protected SparseVector[] za;
    protected int numTokens;
    protected int numTokensChanged;
    protected int numTokensAccepted;
    protected int[][] authorDocIndices; // [A] x [D_a]: store the list of documents for each author
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
            double etaL2,
            double l1,
            double l2,
            boolean mh,
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
        this.etaL2 = etaL2;
        this.l1 = l1;
        this.l2 = l2;
        this.mh = mh;

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
            logln("--- eta l2:\t" + MiscUtils.formatDouble(etaL2));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- MH:\t" + mh);
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
                .append("_el2-").append(formatter.format(etaL2))
                .append("_l1-").append(formatter.format(l1))
                .append("_l2-").append(formatter.format(l2));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_mh-").append(this.mh);
        this.name = str.toString();
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(this.getSamplerFolderPath())
                .append("\nCurrent thread: ").append(Thread.currentThread().getId())
                .append("\n");
        int numNonZerosXs = 0;
        int numXs = 0;
        for (int bb = 0; bb < B; bb++) {
            numNonZerosXs += xy[bb].size();
            numXs += K + 1;
        }
        str.append(">>> # non-zero x's: ").append(numNonZerosXs)
                .append(" / ").append(numXs).append(" (")
                .append(MiscUtils.formatDouble((double) numNonZerosXs / numXs))
                .append(")\n");
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
                if (isValidVote(aa, bb)) {
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
        setupData(docIndices, words, authors, null, authorIndices, null, testVotes);
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
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED, false);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED, false);
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

    /**
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @param testVotes Indicators of test votes
     * @return Predicted probabilities
     */
    public SparseVector[] test(boolean[][] testVotes) {
        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(testVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (testVotes[author][bill]) {
                    double dp = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dp += za[aa].get(kk) * eta[kk] * xy[bb].get(kk);
                    }
                    double score = Math.exp(dp);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
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
            eta[k] = SamplerUtils.getGaussian(0.0, 3.0);
        }

        this.xy = new SparseVector[B];
        for (int bb = 0; bb < B; bb++) {
            if (validBs[bb]) {
                this.xy[bb] = new SparseVector(K + 1);
                for (int kk = 0; kk < K + 1; kk++) {
                    this.xy[bb].set(kk, SamplerUtils.getGaussian(0.0, 3.0));
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

        this.za = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.za[aa] = new SparseVector(K);
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            case PRESET:
                initializePresetAssignments();
                break;
            default:
                throw new RuntimeException("Initialization not supported");
        }
    }

    protected void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED, false);
    }

    protected void initializePresetAssignments() {
        if (verbose) {
            logln("--- Initializing preset assignments. Running LDA ...");
        }

        // run LDA
        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        double lda_alpha = hyperparams.get(ALPHA);
        double lda_beta = hyperparams.get(BETA);

        lda.configure(folder, V, K, lda_alpha, lda_beta, initState, false,
                lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

        int[][] ldaZ = null;
        try {
            File ldaFile = new File(lda.getSamplerFolderPath(), basename + ".zip");
            lda.train(words, null);
            if (ldaFile.exists()) {
                if (verbose) {
                    logln("--- --- LDA file exists. Loading from " + ldaFile);
                }
                lda.inputState(ldaFile);
            } else {
                if (verbose) {
                    logln("--- --- LDA not found. Running LDA ...");
                }
                lda.initialize();
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaFile);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
            ldaZ = lda.getZs();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(log);

        // initialize assignments
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = ldaZ[dd][nn];
                docTopics[dd].increment(z[dd][nn]);
                topicWords[z[dd][nn]].increment(words[dd][nn]);
                za[aa].change(z[dd][nn], authorInversedTokenCounts[aa]);
            }
        }
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
                evaluate();
            }

            updateEtas();
            updateXY();
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED, mh);

            // parameter optimization
            if (iter % LAG == 0 && iter > BURN_IN) {
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
     * Evaluate prediction during training.
     */
    private void evaluate() {
        SparseVector[] predictions = test(validVotes);
        ArrayList<Measurement> measurements = AbstractVotePredictor
                .evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> >>> " + m.getName() + ": " + m.getValue());
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
     * @param isMH
     * @return Elapsed time
     */
    protected long sampleZs(
            boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData,
            boolean observe, boolean isMH) {
        if (isReporting) {
            logln("+++ Sampling assignments ...");
        }
        long sTime = System.currentTimeMillis();
        numTokensChanged = 0;
        numTokensAccepted = 0;

        numTokensChanged = 0;
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                if (removeFromModel) {
                    topicWords[z[dd][nn]].decrement(words[dd][nn]);
                }
                if (removeFromData) {
                    docTopics[dd].decrement(z[dd][nn]);
                    za[aa].change(z[dd][nn], -authorInversedTokenCounts[aa]);
                }

                if (isMH) { // sample using Metropolis-Hastings
                    z[dd][nn] = sampleZMH(dd, nn, observe);
                } else { // sample using Gibbs
                    z[dd][nn] = sampleZGibbs(dd, nn, observe);
                }

                if (addToModel) {
                    topicWords[z[dd][nn]].increment(words[dd][nn]);
                }
                if (addToData) {
                    docTopics[dd].increment(z[dd][nn]);
                    za[aa].change(z[dd][nn], authorInversedTokenCounts[aa]);
                }
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # tokens changed: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                    + ". # tokens accepted: " + numTokensAccepted
                    + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ")");
        }
        return eTime;
    }

    /**
     * Sample a topic for a token by computing the probability of all topics.
     *
     * @param dd
     * @param nn
     * @param observe
     * @return
     */
    private int sampleZGibbs(int dd, int nn, boolean observe) {
        double[] voteLlhs = null;
        if (observe) {
            voteLlhs = getAuthorVoteLogLikelihood(authors[dd]);
        }

        double[] logprobs = new double[K];
        for (int kk = 0; kk < K; kk++) {
            logprobs[kk]
                    = Math.log(docTopics[dd].getCount(kk) + hyperparams.get(ALPHA))
                    + Math.log(topicWords[kk].getProbability(words[dd][nn]));
            if (observe) {
                logprobs[kk] += voteLlhs[kk];
            }
        }
        int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
        numTokensAccepted++;
        if (z[dd][nn] != sampledZ) {
            numTokensChanged++;
        }
        return sampledZ;
    }

    /**
     * Get the log likelihoods of votes by a given voter when assigning a token
     * authored by this voter to each topic.
     *
     * @param aa The voter
     * @return Log likelihoods
     */
    private double[] getAuthorVoteLogLikelihood(int aa) {
        double[] llhs = new double[K];
        for (int bb = 0; bb < B; bb++) {
            if (isValidVote(aa, bb)) {
                double dotprob = xy[bb].get(K);
                for (int kk = 0; kk < K; kk++) {
                    dotprob += za[aa].get(kk) * eta[kk] * xy[bb].get(kk);
                }

                for (int kk = 0; kk < K; kk++) {
                    double propDotProb = dotprob + eta[kk] * xy[bb].get(kk)
                            * authorInversedTokenCounts[aa];
                    llhs[kk] += getVote(aa, bb) * propDotProb
                            - Math.log(1 + Math.exp(propDotProb));
                }
            }
        }
        return llhs;
    }

    /**
     * Sample topic assignment using Metropolis-Hastings.
     *
     * @param dd
     * @param nn
     * @param observe
     * @return
     */
    private int sampleZMH(int dd, int nn, boolean observe) {
        double[] probs = new double[K];
        for (int kk = 0; kk < K; kk++) {
            probs[kk]
                    = (docTopics[dd].getCount(kk) + hyperparams.get(ALPHA))
                    * topicWords[kk].getProbability(words[dd][nn]);
        }
        int sampledZ = SamplerUtils.scaleSample(probs);
        boolean accept;
        if (observe) {
            accept = evaluateProposalAssignment(authors[dd], z[dd][nn], sampledZ);
        } else {
            accept = true;
        }
        if (accept) {
            numTokensAccepted++;
            if (z[dd][nn] != sampledZ) {
                numTokensChanged++;
            }
            return sampledZ;
        } else {
            return z[dd][nn];
        }
    }

    /**
     * Evaluate a proposed assignment and accept/reject the proposal using
     * Metropolis-Hastings ratio.
     *
     * @param aa Author index
     * @param currK Current assignment
     * @param propK Propose assignment
     * @return Accept or reject the proposal
     */
    private boolean evaluateProposalAssignment(int aa, int currK, int propK) {
        double currentLogProb = 0.0;
        double proposalLogProb = 0.0;
        for (int bb = 0; bb < B; bb++) {
            if (isValidVote(aa, bb)) {
                double dotprob = xy[bb].get(K);
                for (int kk = 0; kk < K; kk++) {
                    dotprob += za[aa].get(kk) * eta[kk] * xy[bb].get(kk);
                }

                // current
                double currDotProb = dotprob + eta[currK] * xy[bb].get(currK)
                        * authorInversedTokenCounts[aa];
                currentLogProb += getVote(aa, bb) * currDotProb
                        - Math.log(1 + Math.exp(currDotProb));

                // proposal
                double propDotProb = dotprob + eta[propK] * xy[bb].get(propK)
                        * authorInversedTokenCounts[aa];
                proposalLogProb += getVote(aa, bb) * propDotProb
                        - Math.log(1 + Math.exp(propDotProb));
            }
        }
        double ratio = Math.min(1.0, Math.exp(proposalLogProb - currentLogProb));
        return rand.nextDouble() < ratio;
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

    /**
     * Update X, Y.
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
            minimizer.minimize(xydiff, tempXY, l1);
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
                        dotprod += za[aa].get(kk) * eta[kk] * xy[bb].get(kk);
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
        if (newParams.size() != this.hyperparams.size()) {
            throw new RuntimeException("Number of hyperparameters mismatched");
        }
        double wordLlh = 0.0;
        for (int k = 0; k < K; k++) {
            wordLlh += topicWords[k].getLogLikelihood(newParams.get(BETA) * V,
                    topicWords[k].getCenterVector());
        }

        double topicLlh = 0.0;
        for (int d = 0; d < D; d++) {
            topicLlh += docTopics[d].getLogLikelihood(newParams.get(ALPHA) * K,
                    docTopics[d].getCenterVector());
        }

        double voteLlh = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += za[aa].get(kk) * eta[kk] * xy[bb].get(kk);
                    }
                    voteLlh += getVote(aa, bb) * dotprod - Math.log(1 + Math.exp(dotprod));
                }
            }
        }
        double llh = voteLlh + wordLlh + topicLlh;
        return llh;
    }

    @Override
    public void updateHyperparameters(ArrayList<Double> newParams) {
        this.hyperparams = newParams;
        for (int d = 0; d < D; d++) {
            this.docTopics[d].setConcentration(this.hyperparams.get(ALPHA) * K);
        }
        for (int k = 0; k < K; k++) {
            this.topicWords[k].setConcentration(this.hyperparams.get(BETA) * V);
        }
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
            billStr.append(bb).append("\n");
            billStr.append(SparseVector.output(xy[bb])).append("\n");
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
                int bIdx = Integer.parseInt(reader.readLine());
                if (bIdx != bb) {
                    throw new MismatchRuntimeException(bIdx, bb);
                }
                xy[bb] = SparseVector.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
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
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = xy[bb].get(K);
                        for (int kk = 0; kk < K; kk++) {
                            dotprod += za[aa].get(kk) * xy[bb].get(kk) * w[kk];
                        }
                        llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                    }
                }
            }

            double reg = 0.0;
            for (int kk = 0; kk < K; kk++) {
                reg += 0.5 * etaL2 * w[kk] * w[kk];
            }
            return -llh + reg;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int aa = 0; aa < A; aa++) {
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = xy[bb].get(K);
                        for (int kk = 0; kk < K; kk++) {
                            dotprod += za[aa].get(kk) * xy[bb].get(kk) * w[kk];
                        }
                        double score = Math.exp(dotprod);
                        double prob = score / (1 + score);
                        for (int kk = 0; kk < K; kk++) {
                            grads[kk] -= xy[bb].get(kk) * za[aa].get(kk)
                                    * (getVote(aa, bb) - prob);
                        }
                    }
                }
            }

            for (int kk = 0; kk < K; kk++) {
                grads[kk] += etaL2 * w[kk];
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
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += za[aa].get(kk) * eta[kk] * w[kk];
                    }
                    llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }

            double reg = 0.0;
            if (l2 > 0) {
                for (int ii = 0; ii < w.length; ii++) {
                    reg += 0.5 * l2 * w[ii] * w[ii];
                }
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
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += za[aa].get(kk) * eta[kk] * w[kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] -= za[aa].get(kk) * eta[kk]
                                * (getVote(aa, bb) - prob);
                    }
                    grads[K] -= getVote(aa, bb) - prob;
                }
            }
            if (l2 > 0) {
                for (int kk = 0; kk < w.length; kk++) {
                    grads[kk] += l2 * w[kk];
                }
            }
            return grads;
        }
    }
}
