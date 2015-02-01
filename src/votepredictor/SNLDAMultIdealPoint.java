package votepredictor;

import core.AbstractSampler;
import edu.stanford.nlp.optimization.DiffFunction;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Stack;
import optimization.OWLQN;
import sampler.unsupervised.RecursiveLDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import util.MismatchRuntimeException;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.normalizer.MinMaxNormalizer;

/**
 *
 * @author vietan
 */
public class SNLDAMultIdealPoint extends AbstractSampler {

    protected double[] alphas;          // [L-1]
    protected double[] betas;           // [L]
    protected double[] gamma_means;     // [L-1] mean of bias coins
    protected double[] gamma_scales;    // [L-1] scale of bias coins
    protected double mu;
    protected double[] sigmas;
    protected double l1;
    protected double l2;
    protected double lexl1;
    protected double lexl2;
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
    protected int J; // number of frames per issue
    // derive
    protected int K; // number of issues
    protected int D; // number of documents
    protected int A; // number of authors
    protected int B; // number of bills
    protected int L; // number of levels
    protected boolean[] validAs; // flag voters with no training vote
    protected boolean[] validBs; // flag bills with no training vote
    protected SparseVector[] authorLexDsgMatrix;
    // configuration
    protected PathAssumption pathAssumption;
    protected boolean hasRootTopic;
    // latent
    Node root;
    Node[][] z;
    protected SparseVector[] authorIssueScores; // [A][K]
    protected SparseVector[] xy; // [B][K + 1]
    protected SparseVector[] lexicalParams; // [B][V]
    // internal
    protected SparseVector[] authorBillLexicalScores; // [A][B]
    protected boolean lexReg; // whether performing lexical regression
    protected MinMaxNormalizer[] normalizers;
    protected int numTokens;
    protected int numTokensChanged;
    protected double[] background;
    protected int numTokensAccepted;
    protected int[] authorTokenCounts;  // [A]: store the total #tokens for each author
    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;
    protected ArrayList<String> labelVocab;

    public SNLDAMultIdealPoint() {
        this.basename = "SNLDA-mult-ideal-point";
    }

    public SNLDAMultIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setLabelVocab(ArrayList<String> labelVoc) {
        this.labelVocab = labelVoc;
    }

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public void setVoteVocab(ArrayList<String> voteVoc) {
        this.voteVocab = voteVoc;
    }

    public void configure(String folder,
            int V, int J,
            double[][] issues,
            double[] alphas,
            double[] betas,
            double[] gamma_means,
            double[] gamma_scales,
            double mu, // mean of Gaussian for regression parameters
            double[] sigmas, // variance of Gaussian for regression parameters
            double l1,
            double l2,
            double lexl1, // l1-regularizer for lexical regression
            double lexl2, // l2-regularizer for lexical regression
            boolean hasRootTopic,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.issuePhis = issues;
        this.V = V;
        this.J = J;
        this.K = this.issuePhis.length;
        this.L = 3;

        this.alphas = alphas;
        this.betas = betas;
        this.gamma_means = gamma_means;
        this.gamma_scales = gamma_scales;
        this.mu = mu;
        this.sigmas = sigmas;
        this.l1 = l1;
        this.l2 = l2;
        this.lexl1 = lexl1;
        this.lexl2 = lexl2;
        this.hasRootTopic = hasRootTopic;
        this.lexReg = this.lexl1 > 0 || this.lexl2 > 0;

        this.hyperparams = new ArrayList<Double>();
        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.pathAssumption = pathAssumption;
        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();

        this.setName();

        if (verbose) {
            logln("--- V = " + this.V);
            logln("--- K = " + this.K);
            logln("--- J = " + this.J);
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(gamma_means));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gamma_scales));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2));
            logln("--- lex-l1:\t" + MiscUtils.formatDouble(lexl1));
            logln("--- lex-l2:\t" + MiscUtils.formatDouble(lexl2));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.pathAssumption);
            logln("--- lexical regression?:\t" + lexReg);
        }

        if (this.alphas.length != L - 1) {
            throw new RuntimeException("Local alphas: "
                    + MiscUtils.arrayToString(this.alphas)
                    + ". Length should be " + (L - 1));
        }

        if (this.betas.length != L) {
            throw new RuntimeException("Betas: "
                    + MiscUtils.arrayToString(this.betas)
                    + ". Length should be " + (L));
        }

        if (this.gamma_means.length != L - 1) {
            throw new RuntimeException("Gamma means: "
                    + MiscUtils.arrayToString(this.gamma_means)
                    + ". Length should be " + (L - 1));
        }

        if (this.gamma_scales.length != L - 1) {
            throw new RuntimeException("Gamma scales: "
                    + MiscUtils.arrayToString(this.gamma_scales)
                    + ". Length should be " + (L - 1));
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
                .append("_J-").append(J);
        str.append("_a");
        for (double la : alphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_gm");
        for (double gm : gamma_means) {
            str.append("-").append(MiscUtils.formatDouble(gm));
        }
        str.append("_gs");
        for (double gs : gamma_scales) {
            str.append("-").append(MiscUtils.formatDouble(gs));
        }
        str.append("_m-").append(formatter.format(mu));
        str.append("_ss");
        for (double ss : sigmas) {
            str.append("-").append(MiscUtils.formatDouble(ss));
        }
        str.append("_l1-").append(formatter.format(l1));
        str.append("_l2-").append(formatter.format(l2));
        str.append("_ll1-").append(formatter.format(lexl1));
        str.append("_ll2-").append(formatter.format(lexl2));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_").append(pathAssumption);
        this.name = str.toString();
    }

    protected double getAlpha(int l) {
        return this.alphas[l];
    }

    protected double getBeta(int l) {
        return this.betas[l];
    }

    protected double getSigma(int l) {
        return this.sigmas[l];
    }

    protected double getGammaMean(int l) {
        return this.gamma_means[l];
    }

    protected double getGammaScale(int l) {
        return this.gamma_scales[l];
    }

    /**
     * Flatten the nodes in the tree excluding the root node.
     *
     * @return List of nodes in the tree excluding the root node
     */
    private ArrayList<Node> getNodeList() {
        ArrayList<Node> nodeList = new ArrayList<>();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            if (!node.isRoot()) {
                nodeList.add(node);
            }
        }
        return nodeList;
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath()
                + "\n" + printGlobalTreeSummary()
                + "\nCurrent thread " + Thread.currentThread().getId();
    }

    private int getVote(int aa, int bb) {
        return this.votes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    private boolean isValidVote(int aa, int bb) {
        return this.validVotes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    /**
     * Set training data.
     *
     * @param docIndices Indices of selected documents
     * @param words Document words
     * @param authors Document authors
     * @param votes All votes
     * @param authorIndices Indices of training authors
     * @param billIndices Indices of training bills
     * @param maskedVotes Training votes
     */
    public void setupData(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] maskedVotes) {
        if (verbose) {
            logln("Setting up data ...");
        }

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

        // statistics
        numTokens = 0;
        background = new double[V];
        authorTokenCounts = new int[A];
        for (int dd = 0; dd < D; dd++) {
            numTokens += this.words[dd].length;
            authorTokenCounts[this.authors[dd]] += this.words[dd].length;
            for (int nn = 0; nn <this. words[dd].length; nn++) {
                background[this.words[dd][nn]]++;
            }
        }
        for (int vv = 0; vv < V; vv++) {
            background[vv] /= numTokens;
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

        if (verbose) {
            logln("--- # authors:\t" + A);
            logln("--- # tokens:\t" + numTokens
                    + ", " + StatUtils.sum(authorTokenCounts));
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
     * Make predictions on test data.
     *
     * @return Predictions
     */
    private SparseVector[] makePredictions() {
        if (lexReg) {
            updateAuthorBillLexicalScores();
        }

        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = this.authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < validVotes[author].length; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += authorIssueScores[aa].get(kk) * xy[bb].get(K);
                    }
                    if (lexReg) {
                        dotprod += authorBillLexicalScores[aa].get(bb);
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
        if (lexReg) {
            inputLexicalParameters(stateFile.getAbsolutePath());
        }

        // clear all existing assignments from training data
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.tokenCounts = new SparseCount();
            node.subtreeTokenCounts = new SparseCount();
            node.getContent().clear();
        }

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
            long topicTime;
            if (iter == 0) {
                topicTime = sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED);
            } else {
                topicTime = sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED);
            }

            if (isReporting) {
                logln("--- --- Time. Topic: " + topicTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # tokens changed: " + numTokensChanged
                        + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                        + ". # tokens accepted: " + numTokensAccepted
                        + " (" + MiscUtils.formatDouble((double) numTokensAccepted / numTokens) + ")"
                        + "\n\n");
            }
        }

        if (assignmentFile != null) {
            // assignment string
            StringBuilder assignStr = new StringBuilder();
            for (int dd = 0; dd < z.length; dd++) {
                for (int nn = 0; nn < z[dd].length; nn++) {
                    assignStr.append(dd)
                            .append("\t").append(nn)
                            .append("\t").append(z[dd][nn].getPathString()).append("\n");
                }
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
                        dp += authorIssueScores[aa].get(kk) * xy[bb].get(K);
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
     * Evaluate prediction during training.
     */
    private void evaluate() {
        SparseVector[] predictions = test(validVotes);
        ArrayList<Measurement> measurements = AbstractVotePredictor
                .evaluateAll(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> >>> " + m.getName() + ": " + m.getValue());
        }
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;
        initializeModelStructure();
        initializeDataStructure();
        initializeXY();
        initializeAssignments();
        if (lexReg) {
            initializeLexicalParameters();
        }

        if (debug) {
            validate("Initialized");
        }
        if (verbose) {
            logln("--- Done initializing. \t" + getCurrentState());
            logln("--- " + getLogLikelihood());
        }
    }

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

        // update lexical scores
        updateAuthorBillLexicalScores();

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

    /**
     * Initialize the topics at each node by running LDA recursively.
     */
    protected void initializeModelStructure() {
        int rlda_burnin = 10;
        int rlda_maxiter = 100;
        int rlda_samplelag = 10;
        RecursiveLDA rlda = new RecursiveLDA();
        rlda.setDebug(false);
        rlda.setVerbose(verbose);
        rlda.setLog(false);
        double[] rlda_alphas = {0.25, 0.1};
        double[] rlda_betas = {1.0, 0.1};
        int[] Ks = new int[2];
        Ks[0] = K;
        Ks[1] = J;

        rlda.configure(folder, V, Ks, rlda_alphas, rlda_betas,
                initState, paramOptimized,
                rlda_burnin, rlda_maxiter, rlda_samplelag, rlda_samplelag);
        try {
            File ldaZFile = new File(rlda.getSamplerFolderPath(), basename + ".zip");
            rlda.train(words, null); // words are already filtered using docIndices
            if (ldaZFile.exists()) {
                rlda.inputState(ldaZFile);
            } else {
                rlda.setPriorTopics(issuePhis);
                rlda.initialize();
                rlda.iterate();
                IOUtils.createFolder(rlda.getSamplerFolderPath());
                rlda.outputState(ldaZFile);
                rlda.setWordVocab(wordVocab);
                rlda.outputTopicTopWords(new File(rlda.getSamplerFolderPath(), TopWordFile), 20);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running Recursive LDA for initialization");
        }
        setLog(log);

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, background);
        this.root = new Node(iter, 0, 0, rootTopic, null, 0.0, 0);
        for (int kk = 0; kk < K; kk++) {
            DirMult issueTopic = new DirMult(V, getBeta(1) * V,
                    rlda.getTopicWord(new int[]{kk}).getDistribution());
            Node issueNode = new Node(iter, kk, 1, issueTopic, root,
                    SamplerUtils.getGaussian(mu, getSigma(1)), kk);
            root.addChild(kk, issueNode);
            for (int jj = 0; jj < J; jj++) {
                DirMult frameTopic = new DirMult(V, getBeta(2) * V,
                        rlda.getTopicWord(new int[]{kk, jj}).getDistribution());
                Node frameNode = new Node(iter, jj, 2, frameTopic, issueNode,
                        SamplerUtils.getGaussian(mu, getSigma(2)), kk);
                issueNode.addChild(jj, frameNode);
            }
        }
        this.root.initializeGlobalTheta();
        this.root.initializeGlobalPi();
        for (Node issueNode : this.root.getChildren()) {
            issueNode.initializeGlobalTheta();
            issueNode.initializeGlobalPi();
        }
    }

    protected void initializeDataStructure() {
        z = new Node[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new Node[words[d].length];
        }

        this.authorIssueScores = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorIssueScores[aa] = new SparseVector(K);
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

    protected void initializeXY() {
        this.xy = new SparseVector[B];
        for (int bb = 0; bb < B; bb++) {
            this.xy[bb] = new SparseVector(K + 1);
            for (int kk = 0; kk < K + 1; kk++) {
                this.xy[bb].set(kk, SamplerUtils.getGaussian(0.0, 3.0));
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
                System.out.println();
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
                evaluate();
            }

            updateEtas();
            updateXY();
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);
            if (lexReg) {
                updateLexicalRegression();
            }

            if (debug) {
                validate("iter " + iter);
            }

            // store model
            if (report && iter > BURN_IN && iter % LAG == 0) {
                outputState(new File(reportFolderPath, getIteratedStateFile()));
                outputTopicTopWords(new File(reportFolderPath, getIteratedTopicFile()), 15);
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, getIteratedStateFile()));
            outputTopicTopWords(new File(reportFolderPath, getIteratedTopicFile()), 15);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    /**
     * Add a token to a node.
     *
     * @param nn
     * @param dd
     * @param node
     * @param addToData
     * @param addToModel
     */
    private void addToken(int dd, int nn, Node node,
            boolean addToData, boolean addToModel) {
        if (addToModel) {
            node.getContent().increment(words[dd][nn]);
        }
        if (addToData) {
            int author = authors[dd];
            authorIssueScores[author].change(node.issue, node.eta / authorTokenCounts[author]);

            node.tokenCounts.increment(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeTokenCounts.increment(dd);
                tempNode = tempNode.getParent();
            }
        }
    }

    /**
     * Remove a token from a node.
     *
     * @param nn
     * @param dd
     * @param node
     * @param removeFromData
     * @param removeFromModel
     */
    private void removeToken(int dd, int nn, Node node,
            boolean removeFromData, boolean removeFromModel) {
        if (removeFromData) {
            int author = authors[dd];
            authorIssueScores[author].change(node.issue, -node.eta / authorTokenCounts[author]);

            node.tokenCounts.decrement(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeTokenCounts.decrement(dd);
                tempNode = tempNode.getParent();
            }
        }
        if (removeFromModel) {
            node.getContent().decrement(words[dd][nn]);
        }
    }

    /**
     * Sample node assignment for all tokens.
     *
     * @param removeFromModel
     * @param addToModel
     * @param removeFromData
     * @param addToData
     * @param observe
     * @return Elapsed time
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData, boolean observe) {
        if (isReporting) {
            logln("+++ Sampling assignments ...");
        }

        numTokensChanged = 0;
        numTokensAccepted = 0;
        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                // propose a node
                Node sampledNode = sampleNode(dd, nn, root);
                boolean accept = false;
                if (z[dd][nn] == null) {
                    accept = true;
                } else if (sampledNode.equals(z[dd][nn])) {
                    accept = true;
                    numTokensAccepted++;
                } else {
                    double[] curLogprobs = getLogProbabilities(dd, nn, z[dd][nn], observe);
                    double[] newLogprobs = getLogProbabilities(dd, nn, sampledNode, observe);
                    double ratio = Math.min(1.0,
                            Math.exp(newLogprobs[ACTUAL_INDEX] - curLogprobs[ACTUAL_INDEX]
                                    + curLogprobs[PROPOSAL_INDEX] - newLogprobs[PROPOSAL_INDEX]));
                    if (rand.nextDouble() < ratio) {
                        accept = true;
                        numTokensAccepted++;
                    }
                }

                if (accept) {
                    if (z[dd][nn] != null && !z[dd][nn].equals(sampledNode)) {
                        numTokensChanged++;
                    }
                    z[dd][nn] = sampledNode;
                }

                // add
                addToken(dd, nn, z[dd][nn], addToData, addToModel);
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
     * Recursively sample a node from a current node. The sampled node can be
     * either the same node or one of its children. If the current node is a
     * leaf node, return it.
     *
     * @param dd Document index
     * @param nn Token index
     * @param curNode Current node
     */
    private Node sampleNode(int dd, int nn, Node curNode) {
        if (curNode.isLeaf()) {
            return curNode;
        }
        int level = curNode.getLevel();
        double lAlpha = getAlpha(level);
        double gammaScale = getGammaScale(level);

        double stayprob = 0.0;
        if (hasRootTopic || (!hasRootTopic && !curNode.isRoot())) {
            stayprob = (curNode.tokenCounts.getCount(dd) + gammaScale * curNode.pi)
                    / (curNode.subtreeTokenCounts.getCount(dd) + gammaScale);
        }
        double passprob = 1.0 - stayprob;

        int KK = curNode.getNumChildren();
        double[] probs = new double[KK + 1];
        double norm = curNode.subtreeTokenCounts.getCount(dd)
                - curNode.tokenCounts.getCount(dd) + lAlpha * KK;
        for (Node child : curNode.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * KK * curNode.theta[kk]) / norm;
            double wordprob = child.getPhi(words[dd][nn]);
            probs[kk] = passprob * pathprob * wordprob;
        }
        double wordprob = curNode.getPhi(words[dd][nn]);
        probs[KK] = stayprob * wordprob;

        int sampledIdx = SamplerUtils.scaleSample(probs);
        if (sampledIdx == KK) {
            return curNode;
        } else {
            return sampleNode(dd, nn, curNode.getChild(sampledIdx));
        }
    }

    /**
     * Compute both the proposal log probabilities and the actual log
     * probabilities of assigning a token to a node.
     *
     * @param dd Document index
     * @param nn Token index
     * @param observed
     * @param node The node to be assigned to
     */
    private double[] getLogProbabilities(int dd, int nn, Node node, boolean observed) {
        double[] logprobs = getTransLogProbabilities(dd, nn, node, node);
        logprobs[ACTUAL_INDEX] = Math.log(node.getPhi(words[dd][nn]));
        if (observed) {
            logprobs[ACTUAL_INDEX] += getResponseLogLikelihood(authors[dd], node);
        }
        Node source = node.getParent();
        Node target = node;
        while (source != null) {
            double[] lps = getTransLogProbabilities(dd, nn, source, target);
            logprobs[PROPOSAL_INDEX] += lps[PROPOSAL_INDEX];
            logprobs[ACTUAL_INDEX] += lps[ACTUAL_INDEX];

            source = source.getParent();
            target = target.getParent();
        }
        return logprobs;
    }

    /**
     * Compute the log probabilities of (1) the proposal move and (2) the actual
     * move from source to target. The source and target nodes can be the same.
     *
     * @param dd Document index
     * @param nn Token index
     * @param source The source node
     * @param target The target node
     */
    private double[] getTransLogProbabilities(int dd, int nn, Node source, Node target) {
        int level = source.getLevel();
        if (level == L - 1) { // leaf node
            if (!source.equals(target)) {
                throw new RuntimeException("At leaf node. " + source.toString()
                        + ". " + target.toString());
            }
            return new double[2]; // stay with probabilities 1
        }

        int KK = source.getNumChildren();
        double lAlpha = getAlpha(level);
        double gammaScale = getGammaScale(level);
        double stayprob = (source.tokenCounts.getCount(dd) + gammaScale * source.pi)
                / (source.subtreeTokenCounts.getCount(dd) + gammaScale);
        double passprob = 1.0 - stayprob;

        double pNum = 0.0;
        double pDen = 0.0;
        double aNum = 0.0;
        double aDen = 0.0;
        double norm = source.subtreeTokenCounts.getCount(dd)
                - source.tokenCounts.getCount(dd) + lAlpha * KK;
        for (Node child : source.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * KK * source.theta[kk]) / norm;
            double wordprob = child.getPhi(words[dd][nn]);

            double aVal = passprob * pathprob;
            aDen += aVal;

            double pVal = passprob * pathprob * wordprob;
            pDen += pVal;

            if (target.equals(child)) {
                pNum = pVal;
                aNum = aVal;
            }
        }
        double wordprob = source.getPhi(words[dd][nn]);
        double pVal = stayprob * wordprob;
        pDen += pVal;
        aDen += stayprob;

        if (target.equals(source)) {
            pNum = pVal;
            aNum = stayprob;
        }

        double[] lps = new double[2];
        lps[PROPOSAL_INDEX] = Math.log(pNum / pDen);
        lps[ACTUAL_INDEX] = Math.log(aNum / aDen);
        return lps;
    }

    /**
     * Compute the log likelihood of an author's response variable given that a
     * token from the author is assigned to a given node.
     *
     * @param aa The author
     * @param node The node
     * @return
     */
    private double getResponseLogLikelihood(int aa, Node node) {
        double resLlh = 0.0;
        for (int bb = 0; bb < B; bb++) {
            if (isValidVote(aa, bb)) {
                double dotprod = xy[bb].get(K)
                        + xy[bb].get(node.issue) * node.eta / authorTokenCounts[aa];
                for (int kk : authorIssueScores[aa].getIndices()) {
                    dotprod += authorIssueScores[aa].get(kk) * xy[bb].get(kk);
                }
                if (lexReg) {
                    dotprod += authorBillLexicalScores[aa].get(bb);
                }
                resLlh += getVote(aa, bb) * dotprod - Math.log(1 + Math.exp(dotprod));
            }
        }
        return resLlh;
    }

    /**
     * Update regression parameters using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();

        ArrayList<Node> nodeList = getNodeList();
        double[] etas = new double[nodeList.size()];
        for (int ii = 0; ii < etas.length; ii++) {
            etas[ii] = nodeList.get(ii).eta;
        }

        OWLQN minimizer = new OWLQN();
        minimizer.setQuiet(false);
        minimizer.setMaxIters(100);
        EtaDiffFunc diff = new EtaDiffFunc(nodeList);
        minimizer.minimize(diff, etas, 0.0);
        for (int ii = 0; ii < etas.length; ii++) {
            nodeList.get(ii).eta = etas[ii];
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
            double[] tempXY = xy[bb].dense();
            minimizer.minimize(xydiff, tempXY, 0.0);
            xy[bb] = new SparseVector(tempXY);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
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
        double llh = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += authorIssueScores[aa].get(kk) * xy[bb].get(kk);
                    }
                    if (lexReg) {
                        dotprod += this.authorBillLexicalScores[aa].get(bb);
                    }
                    llh += getVote(aa, bb) * dotprod - Math.log(1 + Math.exp(dotprod));
                }
            }
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
        logln(msg + ". Validation not implemented!");
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
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            modelStr.append(Integer.toString(node.born)).append("\n");
            modelStr.append(Integer.toString(node.issue)).append("\n");
            modelStr.append(node.getPathString()).append("\n");
            modelStr.append(node.eta).append("\n");
            modelStr.append(node.pi).append("\n");
            modelStr.append(SparseCount.output(node.tokenCounts)).append("\n");
            modelStr.append(SparseCount.output(node.subtreeTokenCounts)).append("\n");
            if (node.theta != null) {
                modelStr.append(MiscUtils.arrayToString(node.theta));
            }
            modelStr.append("\n");
            modelStr.append(DirMult.output(node.getContent())).append("\n");
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }

        // assignment string
        StringBuilder assignStr = new StringBuilder();
        for (int dd = 0; dd < z.length; dd++) {
            for (int nn = 0; nn < z[dd].length; nn++) {
                assignStr.append(dd)
                        .append("\t").append(nn)
                        .append("\t").append(z[dd][nn].getPathString()).append("\n");
            }
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

        try { // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());
            contentStrs.add(billStr.toString());
            if (lexReg) {
                contentStrs.add(lexStr.toString());
            }

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + ".bill");
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
            inputBillScore(filepath);
            inputAssignments(filepath);
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

    /**
     * Input a learned model.
     *
     * @param zipFilepath Compressed learned state file
     */
    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);
            HashMap<String, Node> nodeMap = new HashMap<String, Node>();
            String line;
            while ((line = reader.readLine()) != null) {
                int born = Integer.parseInt(line);
                int issue = Integer.parseInt(reader.readLine());
                String pathStr = reader.readLine();
                double eta = Double.parseDouble(reader.readLine());
                double pi = Double.parseDouble(reader.readLine());
                SparseCount tokenCounts = SparseCount.input(reader.readLine());
                SparseCount subtreeTokenCounts = SparseCount.input(reader.readLine());
                line = reader.readLine().trim();
                double[] theta = null;
                if (!line.isEmpty()) {
                    theta = MiscUtils.stringToDoubleArray(line);
                }
                DirMult topic = DirMult.input(reader.readLine());

                // create node
                int lastColonIndex = pathStr.lastIndexOf(":");
                Node parent = null;
                if (lastColonIndex != -1) {
                    parent = nodeMap.get(pathStr.substring(0, lastColonIndex));
                }
                String[] pathIndices = pathStr.split(":");
                int nodeIndex = Integer.parseInt(pathIndices[pathIndices.length - 1]);
                int nodeLevel = pathIndices.length - 1;

                Node node = new Node(born, nodeIndex, nodeLevel, topic, parent, eta, issue);
                node.pi = pi;
                node.theta = theta;
                node.tokenCounts = tokenCounts;
                node.subtreeTokenCounts = subtreeTokenCounts;
                node.setPhiHat(topic.getDistribution());

                if (node.getLevel() == 0) {
                    root = node;
                }
                if (parent != null) {
                    parent.addChild(node.getIndex(), node);
                }
                nodeMap.put(pathStr, node);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    /**
     * Input a set of assignments.
     *
     * @param zipFilepath Compressed learned state file
     */
    public void inputAssignments(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }
        try {
            z = new Node[D][];
            for (int d = 0; d < D; d++) {
                z[d] = new Node[words[d].length];
            }

            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AssignmentFileExt);
            for (int dd = 0; dd < z.length; dd++) {
                for (int nn = 0; nn < z[dd].length; nn++) {
                    String[] sline = reader.readLine().split("\t");
                    if (dd != Integer.parseInt(sline[0])) {
                        throw new MismatchRuntimeException(Integer.parseInt(sline[0]), dd);
                    }
                    if (nn != Integer.parseInt(sline[1])) {
                        throw new MismatchRuntimeException(Integer.parseInt(sline[1]), nn);
                    }
                    String pathStr = sline[2];
                    z[dd][nn] = getNode(pathStr);
                }
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Parse the node path string.
     *
     * @param nodePath The node path string
     * @return
     */
    public int[] parseNodePath(String nodePath) {
        String[] ss = nodePath.split(":");
        int[] parsedPath = new int[ss.length];
        for (int i = 0; i < ss.length; i++) {
            parsedPath[i] = Integer.parseInt(ss[i]);
        }
        return parsedPath;
    }

    /**
     * Get a node in the tree given a parsed path
     *
     * @param parsedPath The parsed path
     */
    private Node getNode(int[] parsedPath) {
        Node node = root;
        for (int i = 1; i < parsedPath.length; i++) {
            node = node.getChild(parsedPath[i]);
        }
        return node;
    }

    private Node getNode(String pathStr) {
        return getNode(parseNodePath(pathStr));
    }

    /**
     * Summary of the current tree.
     *
     * @return Summary of the current tree
     */
    public String printGlobalTreeSummary() {
        StringBuilder str = new StringBuilder();
        SparseCount nodeCountPerLevel = new SparseCount();
        SparseCount obsCountPerLevel = new SparseCount();
        SparseCount subtreeObsCountPerLvl = new SparseCount();
        int numEffNodes = 0;

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);

        int totalObs = 0;
        int totalTokens = 0;
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            int level = node.getLevel();

            if (node.getContent().getCountSum() > 20) {
                numEffNodes++;
            }

            nodeCountPerLevel.increment(level);
            obsCountPerLevel.changeCount(level, node.getContent().getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeTokenCounts.getCountSum());

            totalObs += node.getContent().getCountSum();
            totalTokens += node.tokenCounts.getCountSum();

            for (Node child : node.getChildren()) {
                stack.add(child);
            }
        }
        str.append("global tree:\n\t>>> node count per level:\n");
        for (int l : nodeCountPerLevel.getSortedIndices()) {
            int obsCount = obsCountPerLevel.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLevel.getCount(l);
            str.append("\t>>> >>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
                    .append("]\n");
        }
        str.append("\n");
        str.append("\t>>> # observations = ").append(totalObs).append("\n");
        str.append("\t>>> # tokens = ").append(totalTokens).append("\n");
        str.append("\t>>> # nodes = ").append(nodeCountPerLevel.getCountSum()).append("\n");
        str.append("\t>>> # effective nodes = ").append(numEffNodes);
        return str.toString();
    }

    /**
     * The current tree.
     *
     * @return The current tree
     */
    public String printGlobalTree() {
        SparseCount nodeCountPerLvl = new SparseCount();
        SparseCount obsCountPerLvl = new SparseCount();
        SparseCount subtreeObsCountPerLvl = new SparseCount();
        int totalNumObs = 0;
        int numEffNodes = 0;
        int nodeEffThreshold = 20;

        StringBuilder str = new StringBuilder();
        str.append("global tree\n");

        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            int level = node.getLevel();
            if (node.getContent().getCountSum() > nodeEffThreshold) {
                numEffNodes++;
            }
            if (node.isEmpty()) { // skip empty nodes
                continue;
            }

            nodeCountPerLvl.increment(level);
            obsCountPerLvl.changeCount(level, node.getContent().getCountSum());
            subtreeObsCountPerLvl.changeCount(level, node.subtreeTokenCounts.getCountSum());

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.toString()).append("\n");

            // top words according to distribution
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            String[] topWords = node.getTopWords(10);
            for (String w : topWords) {
                str.append(w).append(" ");
            }
            str.append("\n");

            // top assigned words
            if (!node.getContent().isEmpty()) {
                for (int i = 0; i < node.getLevel(); i++) {
                    str.append("\t");
                }
                str.append(node.getTopObservations()).append("\n");
            }
            str.append("\n");

            totalNumObs += node.getContent().getCountSum();

        }
        str.append("Tree summary").append("\n");
        for (int l : nodeCountPerLvl.getSortedIndices()) {
            int obsCount = obsCountPerLvl.getCount(l);
            int subtreeObsCount = subtreeObsCountPerLvl.getCount(l);
            int nodeCount = nodeCountPerLvl.getCount(l);
            str.append("\t>>> ").append(l)
                    .append(" [")
                    .append(nodeCount)
                    .append("] [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount))
                    .append("] [").append(subtreeObsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) subtreeObsCount / nodeCount))
                    .append("]\n");
        }
        str.append("\t>>> # observations = ").append(totalNumObs).append("\n");
        str.append("\t>>> # nodes = ").append(nodeCountPerLvl.getCountSum()).append("\n");
        str.append("\t>>> # effective nodes = ").append(numEffNodes)
                .append(" (> ").append(nodeEffThreshold).append(")")
                .append("\n");
        return str.toString();
    }

    /**
     * Output top words for each topic in the tree to text file.
     *
     * @param outputFile The output file
     * @param numWords Number of top words
     */
    public void outputTopicTopWords(File outputFile, int numWords) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing top words to file " + outputFile);
        }

        StringBuilder str = new StringBuilder();
        Stack<Node> stack = new Stack<Node>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();

            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> item : rankChildren) {
                stack.add(item.getObject());
            }

            double[] nodeTopic = node.getMLEPhi();
            String[] topWords = getTopWords(nodeTopic, numWords);

            // top words according to the distribution
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.born)
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.eta))
                    .append(")");
            if (node.getLevel() == 1) {
                str.append(" ").append(labelVocab.get(node.getIndex()));
            }
            str.append("\n");

            // words with highest probabilities
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            for (String topWord : topWords) {
                str.append(topWord).append(" ");
            }
            str.append("\n");

            // top assigned words
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getTopObservations()).append("\n\n");
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(str.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topics "
                    + outputFile);
        }
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
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += authorIssueScores[aa].get(kk) * xy[bb].get(kk);
                    }
                    dotprod += authorLexDsgMatrix[aa].dotProduct(w);
                    llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                }
            }

            double val = -llh;
            if (lexl2 > 0) {
                double reg = 0.0;
                for (int ii = 0; ii < w.length; ii++) {
                    reg += 0.5 * lexl2 * w[ii] * w[ii];
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
                    double dotprod = xy[bb].get(K);
                    for (int kk = 0; kk < K; kk++) {
                        dotprod += authorIssueScores[aa].get(kk) * xy[bb].get(kk);
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
                    grads[kk] += lexl2 * w[kk];
                }
            }
            return grads;
        }
    }

    class EtaDiffFunc implements DiffFunction {

        private final ArrayList<Node> nodeList;
        private final SparseVector[] authorEmpNodeDists;

        public EtaDiffFunc(ArrayList<Node> nodeList) {
            this.nodeList = nodeList;
            this.authorEmpNodeDists = new SparseVector[A];
            for (int aa = 0; aa < A; aa++) {
                authorEmpNodeDists[aa] = new SparseVector(nodeList.size());
            }
            for (int kk = 0; kk < nodeList.size(); kk++) {
                Node node = nodeList.get(kk);
                for (int dd : node.tokenCounts.getIndices()) {
                    int count = node.tokenCounts.getCount(dd);
                    int author = authors[dd];
                    double val = (double) count / authorTokenCounts[author];
                    authorEmpNodeDists[author].change(kk, val);
                }
            }
        }

        @Override
        public int domainDimension() {
            return this.nodeList.size();
        }

        @Override
        public double valueAt(double[] w) {
            double llh = 0.0;
            for (int aa = 0; aa < A; aa++) {
                // compute author score at each 1st-level node
                double[] authorScores = new double[K];
                for (int ii = 0; ii < domainDimension(); ii++) {
                    int kk = nodeList.get(ii).issue;
                    authorScores[kk] += authorEmpNodeDists[aa].get(ii) * w[ii];
                }
                // compute llh
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = xy[bb].get(K);
                        for (int kk = 0; kk < K; kk++) {
                            dotprod += authorScores[kk] * xy[bb].get(kk);
                        }
                        if (lexReg) {
                            dotprod += authorBillLexicalScores[aa].get(bb);
                        }
                        llh += getVote(aa, bb) * dotprod - Math.log(Math.exp(dotprod) + 1);
                    }
                }
            }
            // prior
            double reg = 0.0;
            for (int ii = 0; ii < w.length; ii++) {
                reg += 0.5 * w[ii] * w[ii] / getSigma(nodeList.get(ii).getLevel());
            }

            return -llh + reg;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[domainDimension()];
            for (int aa = 0; aa < A; aa++) {
                // compute author score at each 1st-level node
                double[] authorScores = new double[K];
                for (int ii = 0; ii < domainDimension(); ii++) {
                    int kk = nodeList.get(ii).issue;
                    authorScores[kk] += authorEmpNodeDists[aa].get(ii) * w[ii];
                }
                // compute llh
                for (int bb = 0; bb < B; bb++) {
                    if (isValidVote(aa, bb)) {
                        double dotprod = xy[bb].get(K);
                        for (int kk = 0; kk < K; kk++) {
                            dotprod += authorScores[kk] * xy[bb].get(kk);
                        }
                        if (lexReg) {
                            dotprod += authorBillLexicalScores[aa].get(bb);
                        }
                        double score = Math.exp(dotprod);
                        double prob = score / (1 + score);
                        for (int ii = 0; ii < grads.length; ii++) {
                            grads[ii] -= xy[bb].get(nodeList.get(ii).issue)
                                    * authorEmpNodeDists[aa].get(ii) * (getVote(aa, bb) - prob);
                        }
                    }
                }
            }
            // prior
            for (int ii = 0; ii < w.length; ii++) {
                grads[ii] += w[ii] / getSigma(nodeList.get(ii).getLevel());
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
                        dotprod += authorIssueScores[aa].get(kk) * w[kk];
                    }
                    if (lexReg) {
                        dotprod += authorBillLexicalScores[aa].get(bb);
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
                        dotprod += authorIssueScores[aa].get(kk) * w[kk];
                    }
                    if (lexReg) {
                        dotprod += authorBillLexicalScores[aa].get(bb);
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk = 0; kk < K; kk++) {
                        grads[kk] -= authorIssueScores[aa].get(kk) * (getVote(aa, bb) - prob);
                    }
                    grads[K] -= 1.0 * (getVote(aa, bb) - prob);
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

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected final int issue;
        protected SparseCount subtreeTokenCounts;
        protected SparseCount tokenCounts;
        protected double eta; // regression parameter
        protected double pi;
        protected double[] theta;

        // estimated topics after training, which is used for test
        protected double[] phihat;

        public Node(int iter, int index, int level, DirMult content, Node parent,
                double eta, int issue) {
            super(index, level, content, parent);
            this.born = iter;
            this.eta = eta;
            this.issue = issue;
            this.subtreeTokenCounts = new SparseCount();
            this.tokenCounts = new SparseCount();
        }

        void setPhiHat(double[] ph) {
            this.phihat = ph;
        }

        /**
         * Get the probability of a word type given this node. During training,
         * this probability is computed on-the-fly using counts and
         * pseudo-counts. During test, it comes from the learned distribution.
         *
         * @param v word type
         */
        double getPhi(int v) {
            if (this.phihat == null) {
                return getContent().getProbability(v);
            }
            return phihat[v];
        }

        boolean isEmpty() {
            return this.getContent().isEmpty();
        }

        void initializeGlobalPi() {
            this.pi = getGammaMean(level);
        }

        void initializeGlobalTheta() {
            int KK = getNumChildren();
            this.theta = new double[KK];
            Arrays.fill(this.theta, 1.0 / KK);
        }

        double[] getMLEPhi() {
            double[] mapPhi = new double[V];
            for (int vv = 0; vv < V; vv++) {
                mapPhi[vv] = (double) getContent().getCount(vv) / getContent().getCountSum();
            }
            return mapPhi;
        }

        String[] getTopWords(int numTopWords) {
            ArrayList<RankingItem<String>> topicSortedVocab
                    = IOUtils.getSortedVocab(getMLEPhi(), wordVocab);
            String[] topWords = new String[numTopWords];
            for (int i = 0; i < numTopWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
        }

        String getTopObservations() {
            return MiscUtils.getTopObservations(wordVocab, getContent().getSparseCounts(), 10);
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[").append(getPathString());
            str.append(", ").append(born);
            str.append(", ").append(issue);
            str.append(", c (").append(getChildren().size()).append(")");
            str.append(", (").append(getContent().getCountSum()).append(")");
            str.append(", ").append(MiscUtils.formatDouble(eta));
            str.append("]");
            if (this.level == 1 && labelVocab != null) {
                str.append(" [").append(labelVocab.get(index)).append("]")
                        .append(" [")
                        .append(MiscUtils.getTopObservations(wordVocab, issuePhis[index], 7))
                        .append("]");
            }
            return str.toString();
        }
    }
}
