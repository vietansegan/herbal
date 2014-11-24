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
import util.StatUtils;
import util.evaluation.Measurement;
import util.normalizer.MinMaxNormalizer;

/**
 *
 * @author vietan
 */
public class SLDAMultIdealPoint extends AbstractSampler {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public int numSteps = 20; // number of iterations when updating Xs and Ys
    public double rho;
    public double sigma; // eta's variance
    public double gamma; // vote ideal point's variance
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
    protected boolean[] validAs; // flag voters with no training vote
    protected boolean[] validBs; // flag bills with no training vote
    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;
    protected double[] eta; // regression parameters for topics
    protected double[][] xs; // [B][K + 1]
    protected double[][] us; // [A][K]
    // internal
    protected SparseVector[] za;
    protected int numTokens;
    protected int numTokensChanged;
    protected int numTokensAccepted;
    protected int[][] authorDocIndices; // [A] x [D_a]: store the list of documents for each author
    protected double[] authorInversedTokenCounts;
    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;

    // lexical regression
    protected double lexl1; // l1-regularizer for x and y
    protected double lexl2; // l2-regularizer for x and y
    protected boolean lexReg; // whether performing lexical regression
    protected SparseVector[] authorLexDsgMatrix;
    protected SparseVector[] lexicalParams; // [B][V]
    protected SparseVector[] authorBillLexicalScores; // [A][B]
    protected MinMaxNormalizer[] normalizers;

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
            double sigma,
            double gamma,
            double lexl1,
            double lexl2,
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
        this.sigma = sigma;
        this.gamma = gamma;
        this.lexl1 = lexl1;
        this.lexl2 = lexl2;
        this.lexReg = this.lexl1 > 0 || this.lexl2 > 0;

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
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- gamma:\t" + MiscUtils.formatDouble(gamma));
            logln("--- lexical l1:\t" + MiscUtils.formatDouble(lexl1));
            logln("--- lexical l2:\t" + MiscUtils.formatDouble(lexl2));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- lexical regression?:\t" + lexReg);
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
                .append("_s-").append(formatter.format(sigma))
                .append("_g-").append(formatter.format(gamma))
                .append("_ll1-").append(formatter.format(lexl1))
                .append("_ll2-").append(formatter.format(lexl2));
        str.append("_opt-").append(this.paramOptimized);
        this.name = str.toString();
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(this.getSamplerFolderPath())
                .append("\nCurrent thread: ").append(Thread.currentThread().getId())
                .append("\n");
        return str.toString();
    }

    private int getVote(int aa, int bb) {
        return this.votes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    private boolean isValidVote(int aa, int bb) {
        return this.validVotes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    /**
     * Pre-computed statistics.
     */
    protected void prepareDataStatistics() {
        // statistics
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += this.words[d].length;
        }

        // author document list
        ArrayList<Integer>[] authorDocList = new ArrayList[A];
        for (int a = 0; a < A; a++) {
            authorDocList[a] = new ArrayList<Integer>();
        }
        for (int d = 0; d < D; d++) {
            if (this.words[d].length > 0) {
                authorDocList[authors[d]].add(d);
            }
        }
        this.authorDocIndices = new int[A][];
        int[] authorTokenCounts = new int[A];
        for (int a = 0; a < A; a++) {
            this.authorDocIndices[a] = new int[authorDocList[a].size()];
            for (int dd = 0; dd < this.authorDocIndices[a].length; dd++) {
                this.authorDocIndices[a][dd] = authorDocList[a].get(dd);
                authorTokenCounts[a] += this.words[authorDocIndices[a][dd]].length;
            }
        }
        this.authorInversedTokenCounts = new double[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorInversedTokenCounts[aa] = 1.0 / authorTokenCounts[aa];
        }

        if (lexReg) { // compute lexical design matrix if including lexical regression
            this.authorLexDsgMatrix = new SparseVector[A];
            for (int aa = 0; aa < A; aa++) {
                this.authorLexDsgMatrix[aa] = new SparseVector(V);
            }
            for (int dd = 0; dd < D; dd++) {
                int author = this.authors[dd];
                for (int nn = 0; nn < this.words[dd].length; nn++) {
                    this.authorLexDsgMatrix[author].change(this.words[dd][nn], 1.0);
                }
            }
            for (SparseVector sv : this.authorLexDsgMatrix) {
                sv.normalize();
            }
            if (normalizers == null) { // during training
                normalizers = StatUtils.minmaxNormalizeTrainingData(authorLexDsgMatrix, V);
            } else { // during test
                StatUtils.normalizeTestData(authorLexDsgMatrix, normalizers);
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
     * Sample topic assignments for test documents and make predictions.
     *
     * @param stateFile
     * @param predictionFile
     * @param assignmentFile
     * @return
     */
    public SparseVector[] test(File stateFile,
            File predictionFile,
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
        sampleNewDocuments(getFinalStateFile(), assignmentFile);

        if (lexReg) {
            updateAuthorBillLexicalScores();
        }

        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = this.authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < validVotes[author].length; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += za[aa].get(kk) * eta[kk] * xs[bb][kk];
                    }
                    if (lexReg) {
                        dotprod += authorBillLexicalScores[aa].get(bb);
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (score + 1.0);
                    predictions[author].set(bb, prob);
                }
            }
        }
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
        if (lexReg) {
            inputLexicalParameters(stateFile.getAbsolutePath());
        }

        // sample
        for (iter = 0; iter < testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                System.out.println();
                String str = "Iter " + iter + "/" + testMaxIter
                        + "\n" + getCurrentState();
                if (iter < testBurnIn) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
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

                String filename = IOUtils.removeExtension(IOUtils.getFilename(
                        assignmentFile.getAbsolutePath()));
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
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @return Predicted probabilities
     */
    public SparseVector[] testInMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dp = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dp += us[aa][kk] * xs[bb][kk];
                    }
                    if (lexReg) {
                        dp += authorBillLexicalScores[aa].get(bb);
                    }
                    double score = Math.exp(dp);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    /**
     * Make prediction on held-out votes of known legislators and known bills,
     * averaging over multiple models.
     *
     * @return Predictions
     */
    public SparseVector[] testInMatrixMultiples() {
        SparseVector[] predictions = null;
        int count = 0;
        File reportFolder = new File(this.getReportFolderPath());
        String[] files = reportFolder.list();
        for (String file : files) {
            if (!file.endsWith(".zip")) {
                continue;
            }
            this.inputState(new File(reportFolder, file));
            SparseVector[] partPreds = this.testInMatrix();
            if (predictions == null) {
                predictions = partPreds;
            } else {
                for (int aa = 0; aa < predictions.length; aa++) {
                    predictions[aa].add(partPreds[aa]);
                }
            }
            count++;
        }
        for (SparseVector prediction : predictions) {
            prediction.scale(1.0 / count);
        }
        return predictions;
    }

    protected void updateAuthorBillLexicalScores() {
        this.authorBillLexicalScores = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorBillLexicalScores[aa] = new SparseVector(B);
            for (int bb = 0; bb < B; bb++) {
                double dotprod = authorLexDsgMatrix[aa].dotProduct(lexicalParams[bb]);
                this.authorBillLexicalScores[aa].set(bb, dotprod);
            }
        }
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
        if (lexReg) {
            initializeLexicalParameters();
        }

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \n" + getCurrentState());
            getLogLikelihood();
        }
    }

    /**
     * Initialize lexical regression parameters.
     */
    protected void initializeLexicalParameters() {
        File lexregFile = new File(getSamplerFolderPath(), "init-lexreg.txt");
        if (lexregFile.exists()) {
            if (verbose) {
                logln("--- Loading lexical params from " + lexregFile);
            }
            inputInitialLexicalRegression(lexregFile);
        } else {
            if (verbose) {
                logln("--- File not found " + lexregFile);
                logln("--- Running logistic regression ...");
            }
            this.lexicalParams = new SparseVector[B];
            for (int bb = 0; bb < B; bb++) {
                this.lexicalParams[bb] = new SparseVector(V);
            }
            updateLexicalRegression();
            outputInitialLexicalRegression(lexregFile);
        }
        updateAuthorBillLexicalScores(); // update lexical scores
    }

    private void outputInitialLexicalRegression(File outputFile) {
        if (verbose) {
            logln("--- Outputing initial lexical regression to " + outputFile);
        }
        try {
            StringBuilder lexStr = new StringBuilder();
            lexStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                lexStr.append(bb).append("\n");
                lexStr.append(MinMaxNormalizer.output(normalizers[bb])).append("\n");
                lexStr.append(SparseVector.output(lexicalParams[bb])).append("\n");
            }
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(lexStr.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    private void inputInitialLexicalRegression(File inputFile) {
        if (verbose) {
            logln("--- Inputing initial lexical regression from " + inputFile);
        }
        try {
            this.normalizers = new MinMaxNormalizer[B];
            this.lexicalParams = new SparseVector[B];

            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            int numBills = Integer.parseInt(reader.readLine());
            if (numBills != B) {
                throw new MismatchRuntimeException(numBills, B);
            }
            for (int bb = 0; bb < B; bb++) {
                int bIdx = Integer.parseInt(reader.readLine());
                if (bb != bIdx) {
                    throw new MismatchRuntimeException(bIdx, bb);
                }
                this.normalizers[bb] = MinMaxNormalizer.input(reader.readLine());
                this.lexicalParams[bb] = SparseVector.input(reader.readLine());
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + inputFile);
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

        xs = new double[B][K + 1];
        for (int bb = 0; bb < B; bb++) {
            if (validBs[bb]) {
                for (int kk = 0; kk < K + 1; kk++) {
                    xs[bb][kk] = SamplerUtils.getGaussian(0.0, 3.0);
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

        za = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            za[aa] = new SparseVector(K);
        }

        us = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            if (validAs[aa]) {
                for (int kk = 0; kk < K; kk++) {
                    us[aa][kk] = SamplerUtils.getGaussian(0.0, 3.0);
                }
            }
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
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
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
                        //                        + "\t SE = " + getSE()
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }

                // evaluate during training
                SparseVector[] predictions = testInMatrix();
                ArrayList<Measurement> measurements = AbstractVotePredictor
                        .evaluate(votes, validVotes, predictions);
                for (Measurement m : measurements) {
                    logln(">>> >>> " + m.getName() + ": " + m.getValue());
                }
            }

            updateUX();
            updateEtas();
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);
            if (lexReg && iter % REP_INTERVAL == 0) {
                updateLexicalRegression();
            }

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
        numTokensAccepted = 0;
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

                double[] logprobs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    logprobs[kk] = Math.log(docTopics[dd].getCount(kk) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[kk].getProbability(words[dd][nn]));
                    if (observe) {
                        double mean = (za[aa].get(kk) + authorInversedTokenCounts[aa]) * eta[kk];
                        logprobs[kk] += StatUtils.logNormalProbability(us[aa][kk], mean, Math.sqrt(rho));
                    }
                }
                int sampledZ = SamplerUtils.logMaxRescaleSample(logprobs);
                if (z[dd][nn] != sampledZ) {
                    numTokensChanged++;
                }
                z[dd][nn] = sampledZ;

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
            probs[kk] = (docTopics[dd].getCount(kk) + hyperparams.get(ALPHA))
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
                double dotprod = xs[bb][K];
                for (int kk = 0; kk < K; kk++) {
                    dotprod += za[aa].get(kk) * eta[kk] * xs[bb][kk];
                }
                if (lexReg) {
                    dotprod += authorBillLexicalScores[aa].get(bb);
                }

                // current
                double currDotProb = dotprod + eta[currK] * xs[bb][currK]
                        * authorInversedTokenCounts[aa];
                currentLogProb += getVote(aa, bb) * currDotProb
                        - Math.log(1 + Math.exp(currDotProb));

                // proposal
                double propDotProb = dotprod + eta[propK] * xs[bb][propK]
                        * authorInversedTokenCounts[aa];
                proposalLogProb += getVote(aa, bb) * propDotProb
                        - Math.log(1 + Math.exp(propDotProb));
            }
        }
        double ratio = Math.min(1.0, Math.exp(proposalLogProb - currentLogProb));
        return rand.nextDouble() < ratio;
    }

    /**
     * Update eta.
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
//        EtaDiffFunc diff = new EtaDiffFunc();
        EtaDiffNewFunc diff = new EtaDiffNewFunc();
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

    protected long updateUX() {
        if (isReporting) {
            logln("+++ Updating U & X ...");
        }
        long sTime = System.currentTimeMillis();

        for (int ii = 0; ii < numSteps; ii++) {
            updateUs();
            updateXs();
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    protected long updateXs() {
        long sTime = System.currentTimeMillis();
        double bRate = getLearningRate();
        for (int bb = 0; bb < B; bb++) {
            if (!validBs[bb]) {
                continue;
            }
            double[] gradXs = new double[K + 1];
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        gradXs[kk] += us[aa][kk] * (getVote(aa, bb) - prob);
                    }
                    gradXs[K] += getVote(aa, bb) - prob;
                }
            }
            for (int kk = 0; kk < K + 1; kk++) {
                gradXs[kk] -= xs[bb][kk] / gamma;
                xs[bb][kk] += bRate * gradXs[kk];
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        return eTime;
    }

    private long updateUs() {
        long sTime = System.currentTimeMillis();
        double aRate = getLearningRate();
        for (int aa = 0; aa < A; aa++) {
            if (!validAs[aa]) {
                continue;
            }
            double[] grads = new double[K];
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] += xs[bb][kk] * (getVote(aa, bb) - prob);
                    }
                }
            }
            for (int kk = 0; kk < K; kk++) {
                grads[kk] -= (us[aa][kk] - za[aa].get(kk) * eta[kk]) / rho;
                us[aa][kk] += aRate * grads[kk];
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        return eTime;
    }

    public double getLearningRate() {
        return 0.01;
    }

    private long updateLexicalRegression() {
        if (isReporting) {
            logln("+++ Updating lexical regression parameters ...");
        }

        long sTime = System.currentTimeMillis();

        for (int bb = 0; bb < B; bb++) {
            if (isReporting) {
                logln("+++ --- Updating bb = " + bb + " / " + B);
            }
            OWLQN minimizer = new OWLQN();
            minimizer.setQuiet(!isReporting);
            minimizer.setMaxIters(100);
            LexicalDiffFunc diff = new LexicalDiffFunc(bb);
            double[] params = lexicalParams[bb].dense();
            minimizer.minimize(diff, params, lexl1);
            lexicalParams[bb] = new SparseVector(params);
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
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    if (lexReg) {
                        dotprod += authorBillLexicalScores[aa].get(bb);
                    }
                    voteLlh += getVote(aa, bb) * dotprod - Math.log(1 + Math.exp(dotprod));
                }
            }
        }

        double uPrior = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                uPrior += StatUtils.logNormalProbability(us[aa][kk], 
                        za[aa].get(kk) * eta[kk], Math.sqrt(rho));
            }
        }

        double etaPrior = 0.0;
        for (int kk = 0; kk < K; kk++) {
            etaPrior += StatUtils.logNormalProbability(eta[kk], 0.0, Math.sqrt(sigma));
        }

        double xPrior = 0.0;
        for (int bb = 0; bb < B; bb++) {
            for (int kk = 0; kk < K + 1; kk++) {
                xPrior += StatUtils.logNormalProbability(xs[bb][kk], 0.0, Math.sqrt(gamma));
            }
        }

        double llh = voteLlh + wordLlh + topicLlh + uPrior + etaPrior + xPrior;
        if (isReporting) {
            logln("--- --- word-llh: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic-llh: " + MiscUtils.formatDouble(topicLlh)
                    + ". vote-llh: " + MiscUtils.formatDouble(voteLlh)
                    + ". u-prior: " + MiscUtils.formatDouble(uPrior)
                    + ". x-prior: " + MiscUtils.formatDouble(xPrior)
                    + ". eta-prior: " + MiscUtils.formatDouble(etaPrior)
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
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += us[aa][kk] * xs[bb][kk];
                    }
                    if (lexReg) {
                        dotprod += authorBillLexicalScores[aa].get(bb);
                    }
                    voteLlh += getVote(aa, bb) * dotprod - Math.log(1 + Math.exp(dotprod));
                }
            }
        }

        double uPrior = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                uPrior += StatUtils.logNormalProbability(us[aa][kk], 
                        za[aa].get(kk) * eta[kk], Math.sqrt(rho));
            }
        }

        double etaPrior = 0.0;
        for (int kk = 0; kk < K; kk++) {
            etaPrior += StatUtils.logNormalProbability(eta[kk], 0.0, Math.sqrt(sigma));
        }

        double xPrior = 0.0;
        for (int bb = 0; bb < B; bb++) {
            for (int kk = 0; kk < K + 1; kk++) {
                xPrior += StatUtils.logNormalProbability(xs[bb][kk], 0.0, Math.sqrt(gamma));
            }
        }

        double llh = voteLlh + wordLlh + topicLlh + uPrior + etaPrior + xPrior;
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

        // lexical regression
        StringBuilder lexStr = new StringBuilder();
        if (lexReg) {
            lexStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                lexStr.append(bb).append("\n");
                lexStr.append(MinMaxNormalizer.output((MinMaxNormalizer) normalizers[bb]))
                        .append("\n");
                lexStr.append(SparseVector.output(lexicalParams[bb])).append("\n");
            }
        }

        // bills
        StringBuilder billStr = new StringBuilder();
        for (int bb = 0; bb < B; bb++) {
            for (int kk = 0; kk < K + 1; kk++) {
                billStr.append(bb).append("\t").append(kk).append("\t")
                        .append(xs[bb][kk]).append("\n");
            }
        }

        StringBuilder authorStr = new StringBuilder();
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                billStr.append(aa).append("\t").append(kk).append("\t")
                        .append(us[aa][kk]).append("\n");
            }
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
            contentStrs.add(authorStr.toString());
            if (lexReg) {
                contentStrs.add(lexStr.toString());
            }

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + ".bill");
            entryFiles.add(filename + ".author");
            if (lexReg) {
                entryFiles.add(filename + ".lexical");
            }

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
            inputAssignments(filepath);
            inputBillScore(filepath);
            if (lexReg) {
                inputLexicalParameters(filepath);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + filepath);
        }
    }

    public void inputLexicalParameters(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading bill scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ".lexical");
            int numBills = Integer.parseInt(reader.readLine());
            if (B != numBills) {
                throw new MismatchRuntimeException(numBills, B);
            }
            normalizers = new MinMaxNormalizer[B];
            lexicalParams = new SparseVector[B];
            for (int bb = 0; bb < B; bb++) {
                int bIdx = Integer.parseInt(reader.readLine());
                if (bb != bIdx) {
                    throw new MismatchRuntimeException(bIdx, bb);
                }
                normalizers[bb] = MinMaxNormalizer.input(reader.readLine());
                lexicalParams[bb] = SparseVector.input(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
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
                int aa = authors[d];
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                    za[aa].change(z[d][n], authorInversedTokenCounts[aa]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
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

    public void inputBillScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading bill scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ".bill");
            xs = new double[B][K + 1];
            for (int bb = 0; bb < B; bb++) {
                for (int kk = 0; kk < K + 1; kk++) {
                    String[] sline = reader.readLine().split("\t");
                    int bIdx = Integer.parseInt(sline[0]);
                    if (bIdx != bb) {
                        throw new MismatchRuntimeException(bIdx, bb);
                    }
                    int kIdx = Integer.parseInt(sline[1]);
                    if (kIdx != kk) {
                        throw new MismatchRuntimeException(kIdx, kk);
                    }
                    xs[bb][kk] = Double.parseDouble(sline[2]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    public void inputAuthorScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ".author");
            us = new double[A][K];
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    String[] sline = reader.readLine().split("\t");
                    int bIdx = Integer.parseInt(sline[0]);
                    if (bIdx != aa) {
                        throw new MismatchRuntimeException(bIdx, aa);
                    }
                    int kIdx = Integer.parseInt(sline[1]);
                    if (kIdx != kk) {
                        throw new MismatchRuntimeException(kIdx, kk);
                    }
                    us[aa][kk] = Double.parseDouble(sline[2]);
                }
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

    class LexicalDiffFunc implements DiffFunction {

        private final int bb;

        public LexicalDiffFunc(int billIdx) {
            this.bb = billIdx;
        }

        @Override
        public int domainDimension() {
            return V;
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += za[aa].get(kk) * eta[kk] * xs[bb][kk];
                    }
                    dotprod += authorLexDsgMatrix[aa].dotProduct(w);
                    llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }

            double val = -llh;
            if (lexl2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += lexl2 * w[ii] * w[ii];
                }
                val += reg;
            }
            return val;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[w.length];
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xs[bb][K];
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += za[aa].get(kk) * eta[kk] * xs[bb][kk];
                    }
                    dotprod += authorLexDsgMatrix[aa].dotProduct(w);
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int vv = 0; vv < V; vv++) {
                        grads[vv] -= authorLexDsgMatrix[aa].get(vv) * (getVote(aa, bb) - prob);
                    }
                }
            }

            if (lexl2 > 0) {
                for (int kk = 0; kk < w.length; kk++) {
                    grads[kk] += 2 * lexl2 * w[kk];
                }
            }
            return grads;
        }
    }

    /**
     * Optimize eta to make z * eta closer to u.
     */
    class EtaDiffNewFunc implements DiffFunction {

        @Override
        public int domainDimension() {
            return K;
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    double diff = us[aa][kk] - za[aa].get(kk) * w[kk];
                    llh += 0.5 * diff * diff / rho;
                }
            }
            double reg = 0.0;
            for (int kk = 0; kk < K; kk++) {
                reg += 0.5 * w[kk] * w[kk] / sigma;
            }
            return llh + reg;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int kk = 0; kk < K; kk++) {
                for (int aa = 0; aa < A; aa++) {
                    grads[kk] += (za[aa].get(kk) * w[kk] - us[aa][kk]) * za[aa].get(kk) / rho;
                }
                grads[kk] += w[kk] / sigma;
            }
            return grads;
        }
    }

    /**
     * Optimize eta for vote likelihood.
     */
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
                        double dotprod = xs[bb][K];
                        for (int kk = 0; kk < K; kk++) {
                            dotprod += za[aa].get(kk) * xs[bb][kk] * w[kk];
                        }
                        if (lexReg) {
                            dotprod += authorBillLexicalScores[aa].get(bb);
                        }
                        llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                    }
                }
            }

            double reg = 0.0;
            for (int kk = 0; kk < K; kk++) {
                reg += 0.5 * w[kk] * w[kk] / sigma;
            }
            return -llh + reg;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K];
            for (int aa = 0; aa < A; aa++) {
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = xs[bb][K];
                        for (int kk = 0; kk < K; kk++) {
                            dotprod += za[aa].get(kk) * xs[bb][kk] * w[kk];
                        }
                        if (lexReg) {
                            dotprod += authorBillLexicalScores[aa].get(bb);
                        }
                        double score = Math.exp(dotprod);
                        double prob = score / (1 + score);
                        for (int kk = 0; kk < K; kk++) {
                            grads[kk] -= xs[bb][kk] * za[aa].get(kk) * (getVote(aa, bb) - prob);
                        }
                    }
                }
            }

            for (int kk = 0; kk < K; kk++) {
                grads[kk] += w[kk] / sigma;
            }
            return grads;
        }
    }
}
