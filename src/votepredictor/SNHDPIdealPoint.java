package votepredictor;

import cc.mallet.optimize.LimitedMemoryBFGS;
import core.AbstractSampler;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Stack;
import optimization.RidgeLinearRegressionLBFGS;
import sampler.unsupervised.LDA;
import sampler.unsupervised.RecursiveLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
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
public class SNHDPIdealPoint extends AbstractSampler {

    public static final int NEW_CHILD_INDEX = -1;

    protected double[] globalAlphas;
    protected double[] localAlphas;
    protected double[] betas;
    protected double[] gammaMeans; // mean of bias coins
    protected double[] gammaScales; // scale of bias coins
    public double rho;
    public double mu;
    public double[] sigmas;
    public double ipSigma;
    public int numSteps = 20;
    public double epsilon = 0.01;
    // input
    protected int[][] words;
    protected ArrayList<Integer> docIndices;
    protected ArrayList<Integer> authorIndices; // potentially not needed
    protected ArrayList<Integer> billIndices;   // potentially not needed
    protected int[] authors; // [D]: author of each document
    protected int[][] votes;
    protected boolean[][] trainVotes;
    protected double[][] issuePhis;
    protected int V; // vocabulary size
    // derive
    protected int K; // number of issues
    protected int D; // number of documents
    protected int A; // number of authors
    protected int B; // number of bills
    protected int L = 3; // number of levels
    // configuration
    protected PathAssumption pathAssumption;
    protected boolean hasRootTopic;
    // latent
    Node root;
    Node[][] z;
    protected double[] u; // [A]: authors' scores
    protected double[] x; // [B]
    protected double[] y; // [B]
    protected double[] authorMeans;
    // internal
    protected int numTokens;
    protected int numTokensChanged;
    protected double[] background;
    protected int numTokensAccepted;
    protected int[][] authorDocIndices; // [A] x [D_a]: store the list of documents for each author
    protected int[] authorTokenCounts;  // [A]: store the total #tokens for each author
    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;
    protected ArrayList<String> labelVocab;
    protected int posAnchor;
    protected int negAnchor;

    public SNHDPIdealPoint() {
        this.basename = "SNHDP-ideal-point";
    }

    public SNHDPIdealPoint(String bname) {
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

    protected double getSigma(int l) {
        return this.sigmas[l];
    }

    protected double getLocalAlpha(int l) {
        return this.localAlphas[l];
    }

    protected double getGlobalAlpha(int l) {
        return this.globalAlphas[l];
    }

    protected double getBeta(int l) {
        return this.betas[l];
    }

    protected double getGammaMean(int l) {
        return this.gammaMeans[l];
    }

    protected double getGammaScale(int l) {
        return this.gammaScales[l];
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

    public void configure(SNHDPIdealPoint sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.issuePhis,
                sampler.globalAlphas,
                sampler.localAlphas,
                sampler.betas,
                sampler.gammaMeans,
                sampler.gammaScales,
                sampler.rho,
                sampler.mu,
                sampler.sigmas,
                sampler.ipSigma,
                sampler.hasRootTopic,
                sampler.initState,
                sampler.pathAssumption,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V,
            double[][] issues,
            double[] global_alphas,
            double[] local_alphas,
            double[] betas,
            double[] gamma_means,
            double[] gamma_scales,
            double rho,
            double mu, // mean of Gaussian for regression parameters
            double[] sigmas, // stadard deviation of Gaussian for regression parameters
            double ipSigma,
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
        this.K = this.issuePhis.length;

        this.globalAlphas = global_alphas;
        this.localAlphas = local_alphas;
        this.betas = betas;
        this.gammaMeans = gamma_means;
        this.gammaScales = gamma_scales;
        this.rho = rho;
        this.mu = mu;
        this.sigmas = sigmas;
        this.ipSigma = ipSigma;
        this.hasRootTopic = hasRootTopic;

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
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(local_alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- gamma means:\t" + MiscUtils.arrayToString(gamma_means));
            logln("--- gamma scales:\t" + MiscUtils.arrayToString(gamma_scales));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- ideal point sigma:\t" + MiscUtils.formatDouble(ipSigma));
            logln("--- has root topic:\t" + hasRootTopic);
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.pathAssumption);
        }

        if (this.localAlphas.length != this.betas.length - 1) {
            throw new RuntimeException("Lengths mismatch.");
        }
    }

    protected void setName() {
        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG);
        str.append("_a");
        for (double la : localAlphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_gm");
        for (double gm : gammaMeans) {
            str.append("-").append(MiscUtils.formatDouble(gm));
        }
        str.append("_gs");
        for (double gs : gammaScales) {
            str.append("-").append(MiscUtils.formatDouble(gs));
        }
        str.append("_r-").append(formatter.format(rho))
                .append("_m-").append(formatter.format(mu));
        str.append("_ss");
        for (double ss : sigmas) {
            str.append("-").append(MiscUtils.formatDouble(ss));
        }
        str.append("_s-").append(formatter.format(ipSigma));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_rt-").append(this.hasRootTopic);
        str.append("_").append(pathAssumption);
        this.name = str.toString();
    }

    /**
     * Pre-compute useful data statistics.
     */
    protected void prepareDataStatistics() {
        // statistics
        numTokens = 0;
        background = new double[V];
        for (int dd = 0; dd < D; dd++) {
            numTokens += words[dd].length;
            for (int nn = 0; nn < words[dd].length; nn++) {
                background[words[dd][nn]]++;
            }
        }
        for (int vv = 0; vv < V; vv++) {
            background[vv] /= numTokens;
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
     * @param authorIndices Indices of training authors
     * @param billIndices Indices of training bills
     * @param trainVotes Training votes
     */
    public void train(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
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

        // votes and training votes
        this.votes = new int[A][B];
        this.trainVotes = new boolean[A][B];
        for (int ii = 0; ii < A; ii++) {
            int aa = this.authorIndices.get(ii);
            for (int jj = 0; jj < B; jj++) {
                int bb = this.billIndices.get(jj);
                this.votes[ii][jj] = votes[aa][bb];
                this.trainVotes[ii][jj] = trainVotes[aa][bb];
            }
        }

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

                    if (score == 0) {
                        System.out.println("aa = " + aa + ". " + u[aa]);
                        System.out.println("bb = " + bb + ". " + x[bb] + ". " + y[bb]);
                        System.out.println("raw: " + u[aa] * x[bb] + y[bb]);
                        throw new RuntimeException();
                    }

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
            File partVoteScoreFile) {
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

        this.setTestConfigurations(50, 100, 5, 5);

        this.sampleNewDocuments(getFinalStateFile(), words, docIndices);

        // predict author scores
        int testA = authorIndices.size();
        double[] predAuthorScores = new double[testA];
        double[] predAuthorDens = new double[testA];
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            for (int ii : node.tokenCounts.getIndices()) {
                int count = node.tokenCounts.getCount(ii);
                int author = authors[docIndices.get(ii)];
                int aa = authorIndices.indexOf(author);
                if (aa < 0) {
                    throw new RuntimeException("aa = " + aa + ". " + author);
                }
                predAuthorScores[aa] += count * node.eta;
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

        if (predictionFile != null) { // output predictions
            AbstractVotePredictor.outputPredictions(predictionFile, null, predictions);
        }

        if (partAuthorScoreFile != null) { // output author scores
            AbstractVotePredictor.outputAuthorScores(partAuthorScoreFile, null, u);
        }

        if (partVoteScoreFile != null) { // output vote scores
            AbstractVotePredictor.outputVoteScores(partVoteScoreFile, null, x, y);
        }

        return predictions;
    }

    /**
     * Sample topic assignments for all tokens in a set of test documents.
     *
     * @param stateFile
     * @param testWords
     * @param testDocIndices
     */
    private void sampleNewDocuments(
            File stateFile,
            int[][] testWords,
            ArrayList<Integer> testDocIndices) {
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

        if (verbose) {
            logln("--- # test documents: " + docIndices.size());
            logln("--- # tokens: " + this.numTokens);
        }

        z = new Node[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new Node[words[d].length];
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
                topicTime = sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED, !EXTEND);
            } else {
                topicTime = sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED, !EXTEND);
            }

            if (disp) {
                logln("--- --- Time. Topic: " + topicTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # tokens changed: " + numTokensChanged
                        + " (" + (double) numTokensChanged / numTokens + ")"
                        + ". # tokens accepted: " + numTokensAccepted
                        + " (" + (double) numTokensAccepted / numTokens + ")"
                        + "\n\n");
            }
        }
    }

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        initializeModelStructureOneLevel();
        initializeDataStructure();
        initializeUXY();
        initializeAssignments();
        updateEtas();
        if (debug) {
            validate("Initialized");
        }
        if (verbose) {
            logln("--- Done initializing. \t" + getCurrentState());
            logln("\n" + printGlobalTree() + "\n\n");
        }
    }

    /**
     * Run LDA to initialize the first-level nodes.
     */
    protected void initializeModelStructureOneLevel() {
        // run LDA
        int lda_burnin = 10;
        int lda_maxiter = 100;
        int lda_samplelag = 10;
        LDA lda = new LDA();
        lda.setDebug(debug);
        lda.setVerbose(verbose);
        lda.setLog(false);
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        lda.configure(folder, V, K, lda_alpha, lda_beta, initState, paramOptimized,
                lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);
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
                lda.initialize(null, issuePhis);
                lda.iterate();
                IOUtils.createFolder(lda.getSamplerFolderPath());
                lda.outputState(ldaFile);
                lda.setWordVocab(wordVocab);
                lda.outputTopicTopWords(new File(lda.getSamplerFolderPath(), TopWordFile), 20);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(log);

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, background);
        this.root = new Node(iter, 0, 0, false, rootTopic, null, 0.0);
        for (int kk = 0; kk < K; kk++) {
            DirMult issueTopic = new DirMult(V, getBeta(1) * V,
                    lda.getTopicWords()[kk].getDistribution());
            Node issueNode = new Node(iter, kk, 1, true, issueTopic, root,
                    SamplerUtils.getGaussian(mu, getSigma(1)));
            root.addChild(kk, issueNode);
        }

        // initialize theta and pi
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            node.changeStatus();
            if (node.getLevel() < L - 1) {
                for (Node child : node.getChildren()) {
                    stack.add(child);
                }
                node.initializeGlobalTheta();
                node.initializeGlobalPi();
            }
        }
        x = new double[B];
        y = new double[B];
        u = new double[A];
    }

    /**
     * Run Recursive LDA to initialize nodes in two levels.
     */
    protected void initializeModelStructureTwoLevels() {
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
        Ks[1] = 10;

        rlda.configure(folder, V, Ks, rlda_alphas, rlda_betas,
                initState, paramOptimized,
                rlda_burnin, rlda_maxiter, rlda_samplelag, rlda_samplelag);
        try {
            File rldaFile = new File(rlda.getSamplerFolderPath(), basename + ".zip");
            rlda.train(words, null); // words are already filtered using docIndices
            if (rldaFile.exists()) {
                if (verbose) {
                    logln("--- RecursiveLDA file exists. Loading from " + rldaFile);
                }
                rlda.inputState(rldaFile);
            } else {
                if (verbose) {
                    logln("--- RecursiveLDA not exists. Running ... ");
                }
                rlda.setPriorTopics(issuePhis);
                rlda.initialize();
                rlda.iterate();
                rlda.setWordVocab(wordVocab);

                if (verbose) {
                    logln("--- --- Outputing to " + rldaFile);
                }
                IOUtils.createFolder(rlda.getSamplerFolderPath());
                rlda.outputState(rldaFile);
                rlda.outputTopicTopWords(new File(rlda.getSamplerFolderPath(), TopWordFile), 20);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running Recursive LDA for initialization");
        }
        setLog(log);

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, background);
        this.root = new Node(iter, 0, 0, false, rootTopic, null, 0.0);
        for (int kk = 0; kk < K; kk++) {
            DirMult issueTopic = new DirMult(V, getBeta(1) * V,
                    rlda.getTopicWord(new int[]{kk}).getDistribution());
            Node issueNode = new Node(iter, kk, 1, true, issueTopic, root,
                    SamplerUtils.getGaussian(mu, getSigma(1)));
            root.addChild(kk, issueNode);
            for (int jj = 0; jj < Ks[1]; jj++) {
                DirMult frameTopic = new DirMult(V, getBeta(2) * V,
                        rlda.getTopicWord(new int[]{kk, jj}).getDistribution());
                Node frameNode = new Node(iter, jj, 2, false, frameTopic, issueNode,
                        SamplerUtils.getGaussian(mu, getSigma(2)));
                issueNode.addChild(jj, frameNode);
            }
        }

        // initialize theta and pi
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            node.changeStatus();
            if (node.getLevel() < L - 1) {
                for (Node child : node.getChildren()) {
                    stack.add(child);
                }
                node.initializeGlobalTheta();
                node.initializeGlobalPi();
            }
        }
        x = new double[B];
        y = new double[B];
        u = new double[A];
    }

    protected void initializeDataStructure() {
        z = new Node[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new Node[words[d].length];
        }
        authorMeans = new double[A];
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
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED, EXTEND);

        // remove empty nodes after initial assignments and update thetas
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();

            if (node.getLevel() < L - 1) {
                ArrayList<Node> emptyChildren = new ArrayList<>();
                for (Node child : node.getChildren()) {
                    if (child.isEmpty()) {
                        emptyChildren.add(child);
                    } else {
                        stack.add(child);
                    }
                }

                for (Node emptyChild : emptyChildren) {
                    node.removeChild(emptyChild.getIndex());
                }
                node.updateGlobalTheta();
            }
        }
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
                if (trainVotes[aa][bb]) {
                    if (votes[aa][bb] == Vote.WITH) {
                        withCount++;
                    } else if (votes[aa][bb] == Vote.AGAINST) {
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
                this.u[aa] = SamplerUtils.getGaussian(mu, ipSigma);
            }
        }

        this.x = new double[B];
        this.y = new double[B];
        for (int b = 0; b < B; b++) {
            this.x[b] = SamplerUtils.getGaussian(mu, ipSigma);
            this.y[b] = SamplerUtils.getGaussian(mu, ipSigma);
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
            boolean isReporting = isReporting();
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
            }

            long topicTime = sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED, EXTEND);
            long thetaTime = updateThetas();
            long etaTime = updateEtas();
            long uxyTime = updateUXY();

            if (isReporting) {
                logln("--- --- Time. Topic: " + topicTime
                        + ". Theta: " + thetaTime
                        + ". Eta: " + etaTime
                        + ". UXY: " + uxyTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # tokens changed: " + numTokensChanged
                        + " (" + (double) numTokensChanged / numTokens + ")"
                        + ". # tokens accepted: " + numTokensAccepted
                        + " (" + (double) numTokensAccepted / numTokens + ")");
                if (debug) {
                    logln("\n" + printGlobalTree() + "\n\n");
                } else {
                    logln("\n" + printGlobalTreeSummary() + "\n\n");
                }
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
            if (authorMeans != null) { // don't update during test
                authorMeans[authors[dd]] += node.eta / authorTokenCounts[authors[dd]];
            }
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
            if (authorMeans != null) { // don't update during test
                authorMeans[authors[dd]] -= node.eta / authorTokenCounts[authors[dd]];
            }
            node.tokenCounts.decrement(dd);
            Node tempNode = node;
            while (tempNode != null) {
                tempNode.subtreeTokenCounts.decrement(dd);
                tempNode = tempNode.getParent();
            }
        }

        if (removeFromModel) {
            node.getContent().decrement(words[dd][nn]);

            if (node.subtreeTokenCounts.isEmpty()) {
                if (!node.tokenCounts.isEmpty()) {
                    throw new RuntimeException("SubtreeTokenCounts is empty"
                            + " but TokenCounts is not.\n" + node.toString());
                }
                Node tempNode = node;
                while (tempNode.subtreeTokenCounts.isEmpty()) {
                    Node parent = tempNode.getParent();
                    parent.removeChild(tempNode.getIndex());
                    parent.updateGlobalTheta();
                    tempNode = parent;
                }
            }
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
     * @param extend
     * @return Elapsed time
     */
    protected long sampleZs(boolean removeFromModel, boolean addToModel,
            boolean removeFromData, boolean addToData, boolean observe,
            boolean extend) {
        numTokensChanged = 0;
        numTokensAccepted = 0;
        long sTime = System.currentTimeMillis();
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

                Node sampledNode = sampleNode(dd, nn, root, extend);
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
                    } else {
                        // if a new node is sampled and rejected, remove it 
                        if (sampledNode.newNode) {
                            sampledNode.getParent().removeChild(sampledNode.getIndex());
                        }
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

                Node parent = z[dd][nn].getParent();
                int zIdx = z[dd][nn].getIndex();
                if (accept) {
                    // if a new node is sampled and accepted, change its status
                    // (not new node anymore) and udpate the global theta of its parent
                    if (z[dd][nn].newNode) {
                        z[dd][nn].changeStatus();
                        parent.updateGlobalTheta();
                    }
                } else {
                    // if reject the proposed node and the current node is removed
                    // from the tree, we need to add it back to the tree
                    if (!z[dd][nn].isRoot() && !parent.hasChild(zIdx)) {
                        parent.addChild(zIdx, z[dd][nn]);
                        parent.updateGlobalTheta();
                    }
                }
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Recursively sample a node from a current node. The sampled node can be
     * either the same node, one of its children or a new child node.
     *
     * @param dd Document index
     * @param nn Token index
     * @param curNode Current node
     * @param extend Whether extensible or not
     * @return A sampled node
     */
    private Node sampleNode(int dd, int nn, Node curNode, boolean extend) {
        int level = curNode.getLevel();
        if (level == L - 1) {
            return curNode;
        }
        double lAlpha = getLocalAlpha(level);
        double gammaScale = getGammaScale(level);

        double stayprob = 0.0;
        if (hasRootTopic || (!hasRootTopic && !curNode.isRoot())) {
            stayprob = (curNode.tokenCounts.getCount(dd) + gammaScale * curNode.pi)
                    / (curNode.subtreeTokenCounts.getCount(dd) + gammaScale);
        }
        double passprob = 1.0 - stayprob;

        ArrayList<Node> nodeList = new ArrayList<>();
        ArrayList<Double> nodeProbs = new ArrayList<>();
        double norm = curNode.subtreeTokenCounts.getCount(dd)
                - curNode.tokenCounts.getCount(dd) + lAlpha;

        // existing children
        for (Node child : curNode.getChildren()) {
            int childIdx = child.getIndex();
            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * curNode.theta.get(childIdx)) / norm;
            double wordprob = child.getPhi(words[dd][nn]);

            nodeList.add(child);
            nodeProbs.add(passprob * pathprob * wordprob);
        }

        // stay put
        double stayWordProb = curNode.getPhi(words[dd][nn]);
        nodeList.add(curNode);
        nodeProbs.add(stayprob * stayWordProb);

        // new children
        if (extend && curNode.isExtensible()) {
            double newWordProb = 1.0 / V;
            double newPathProb = lAlpha * curNode.theta.get(NEW_CHILD_INDEX) / norm;
            nodeList.add(null);
            nodeProbs.add(passprob * newPathProb * newWordProb);
        }

        int sampledIdx = SamplerUtils.scaleSample(nodeProbs);
        Node sampledNode = nodeList.get(sampledIdx);
        if (sampledNode == null) {
            int newChildIdx = curNode.getNextChildIndex();
            Node newChild = new Node(iter, newChildIdx, level + 1, false,
                    new DirMult(V, getBeta(level + 1) * V, 1.0 / V), curNode,
                    SamplerUtils.getGaussian(mu, getSigma(level + 1)));
            curNode.addChild(newChildIdx, newChild);
            return newChild;
        } else if (sampledNode.equals(curNode)) {
            return curNode;
        } else {
            return sampleNode(dd, nn, sampledNode, extend);
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
     * Get the transition log probability of moving a token from a source node
     * to a target node. The source node can be the same as the target node.
     *
     * @param dd
     * @param nn
     * @param source
     * @param target
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

        double pNum = 0.0;
        double pDen = 0.0;
        double aNum = 0.0;
        double aDen = 0.0;

        double lAlpha = getLocalAlpha(level);
        double gammaScale = getGammaScale(level);
        double stayprob = (source.tokenCounts.getCount(dd) + gammaScale * source.pi)
                / (source.subtreeTokenCounts.getCount(dd) + gammaScale);
        double passprob = 1.0 - stayprob;
        double norm = source.subtreeTokenCounts.getCount(dd)
                - source.tokenCounts.getCount(dd) + lAlpha;

        // existing children
        boolean foundTarget = false;
        for (Node child : source.getChildren()) {
            double pathprob;
            double wordprob;
            if (child.newNode) { // newly created child
                wordprob = 1.0 / V;
                pathprob = lAlpha * source.theta.get(NEW_CHILD_INDEX) / norm;
            } else {
                try {
                    pathprob = (child.subtreeTokenCounts.getCount(dd)
                            + lAlpha * source.theta.get(child.getIndex())) / norm;
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("source: " + source.toString());
                    System.out.println("target: " + target.toString());
                    System.out.println("child: " + child.toString());
                    System.out.println("subtree: " + child.subtreeTokenCounts.getCount(dd));
                    System.out.println("theta: " + source.theta.get(child.getIndex()));
                    throw new RuntimeException("Exception");
                }

                wordprob = child.getPhi(words[dd][nn]);
            }

            double aVal = passprob * pathprob;
            aDen += aVal;

            double pVal = passprob * pathprob * wordprob;
            pDen += pVal;

            if (target.equals(child)) { // including a new child
                pNum = pVal;
                aNum = aVal;
                foundTarget = true;
            }
        }

        // staying at the current node
        double wordprob = source.getPhi(words[dd][nn]);
        double pVal = stayprob * wordprob;
        pDen += pVal;
        aDen += stayprob;

        if (target.equals(source)) {
            pNum = pVal;
            aNum = stayprob;
            foundTarget = true;
        }

        if (!foundTarget) {
            if (!target.isEmpty()) {
                throw new RuntimeException("Target node is not empty and could not be found");
            }

            double wProb = 1.0 / V;
            double pProb = lAlpha * source.theta.get(NEW_CHILD_INDEX) / norm;
            double aVal = passprob * pProb;
            aDen += aVal;
            pVal = passprob * pProb * wProb;
            pDen += pVal;

            pNum = pVal;
            aNum = aVal;
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
     * @param author The author
     * @param node The node
     * @return
     */
    private double getResponseLogLikelihood(int author, Node node) {
        double aMean = authorMeans[author] + node.eta / authorTokenCounts[author];
        double resLLh = StatUtils.logNormalProbability(u[author], aMean, Math.sqrt(rho));
        return resLLh;
    }

    /**
     * Update regression parameters using L-BFGS.
     *
     * @return Elapsed time
     */
    public long updateEtas() {
        long sTime = System.currentTimeMillis();

        // list of nodes
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        // design matrix
        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(K);
        }
        for (int kk = 0; kk < N; kk++) {
            Node node = nodeList.get(kk);
            for (int dd : node.tokenCounts.getIndices()) {
                int count = node.tokenCounts.getCount(dd);
                int author = authors[dd];
                double val = (double) count / authorTokenCounts[author];
                designMatrix[author].change(kk, val);
            }
        }

        // current params
        double[] nodeSigmas = new double[N];
        double[] etas = new double[N];
        for (int kk = 0; kk < N; kk++) {
            etas[kk] = nodeList.get(kk).eta;
            nodeSigmas[kk] = getSigma(nodeList.get(kk).getLevel());
        }

        RidgeLinearRegressionLBFGS optimizable = new RidgeLinearRegressionLBFGS(
                u, etas, designMatrix, rho, mu, nodeSigmas);

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
        for (int kk = 0; kk < N; kk++) {
            nodeList.get(kk).eta = optimizable.getParameter(kk);
        }
        // update author means
        for (int aa = 0; aa < A; aa++) {
            authorMeans[aa] = 0.0;
            for (int kk : designMatrix[aa].getIndices()) {
                authorMeans[aa] += designMatrix[aa].get(kk) * nodeList.get(kk).eta;
            }
        }
        return System.currentTimeMillis() - sTime;
    }

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

    /**
     * Update the global thetas.
     *
     * @return Elapsed time
     */
    private long updateThetas() {
        long sTime = System.currentTimeMillis();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            if (node.getLevel() < L - 1) {
                for (Node child : node.getChildren()) {
                    stack.add(child);
                }
                node.updateGlobalTheta();
            }
        }
        return System.currentTimeMillis() - sTime;
    }

    /**
     * Update U, X and Y.
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
            // likelihood
            for (int b = 0; b < votes[a].length; b++) {
                if (trainVotes[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    grad += x[b] * (votes[a][b] - prob); // only work for 0 and 1
                }
            }
            // prior
            grad -= (u[a] - authorMeans[a]) / ipSigma;
            // update
            u[a] += epsilon * grad;
        }
    }

    public void updateXYs() {
        for (int b = 0; b < B; b++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int a = 0; a < A; a++) {
                if (trainVotes[a][b]) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    gradX += u[a] * (votes[a][b] - score / (1 + score));
                    gradY += votes[a][b] - score / (1 + score);
                }
            }
            // prior
            gradX -= (x[b] - mu) / ipSigma;
            gradY -= (y[b] - mu) / ipSigma;
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
            if (trainVotes[author][b]) {
                double score = authorVal * x[b] + y[b];
                llh += votes[author][b] * score - Math.log(1 + Math.exp(score));
            }
        }
        return llh;
    }

    @Override
    public double getLogLikelihood() {
        return getVoteLogLikelihood();
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
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.validate(msg);
        }
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }
        // model string
        StringBuilder modelStr = new StringBuilder();

        // authors
        modelStr.append(A).append("\n");
        for (int aa = 0; aa < A; aa++) {
            modelStr.append(u[aa]).append("\n");
        }

        // bills
        modelStr.append(B).append("\n");
        for (int bb = 0; bb < B; bb++) {
            modelStr.append(x[bb]).append("\t").append(y[bb]).append("\n");
        }

        // tree
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            modelStr.append(Integer.toString(node.born)).append("\n");
            modelStr.append(node.getPathString()).append("\n");
            modelStr.append(node.eta).append("\n");
            modelStr.append(node.pi).append("\n");
            modelStr.append(SparseCount.output(node.tokenCounts)).append("\n");
            modelStr.append(SparseCount.output(node.subtreeTokenCounts)).append("\n");
            if (node.theta != null) {
                modelStr.append(hashMapToString(node.theta));
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

        // output to a compressed file
        try {
            this.outputZipFile(filepath, modelStr.toString(), assignStr.toString());
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
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + filepath);
        }
    }

    /**
     * Input a learned model.
     *
     * @param zipFilepath Compressed learned state file
     */
    void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ModelFileExt);

            A = Integer.parseInt(reader.readLine());
            u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                u[aa] = Double.parseDouble(reader.readLine());
            }

            B = Integer.parseInt(reader.readLine());
            x = new double[B];
            y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                String[] sline = reader.readLine().split("\t");
                x[bb] = Double.parseDouble(sline[0]);
                y[bb] = Double.parseDouble(sline[1]);
            }

            HashMap<String, Node> nodeMap = new HashMap<String, Node>();
            String line;
            while ((line = reader.readLine()) != null) {
                int born = Integer.parseInt(line);
                String pathStr = reader.readLine();
                double eta = Double.parseDouble(reader.readLine());
                double pi = Double.parseDouble(reader.readLine());
                SparseCount tokenCounts = SparseCount.input(reader.readLine());
                SparseCount subtreeTokenCounts = SparseCount.input(reader.readLine());
                line = reader.readLine().trim();
                HashMap<Integer, Double> theta = new HashMap<>();
                if (!line.isEmpty()) {
                    theta = stringToHashMap(line);
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

                Node node = new Node(born, nodeIndex, nodeLevel, false, topic, parent, eta);
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

        if (verbose && debug) {
            logln(printGlobalTree());
        }
    }

    /**
     * Input a set of assignments.
     *
     * @param zipFilepath Compressed learned state file
     */
    void inputAssignments(String zipFilepath) throws Exception {
        if (verbose) {
            logln("--- --- Loading assignments from " + zipFilepath);
        }
        System.out.println("--- --- --- TO BE IMPLEMENTED");
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
            if (node.getContent().getCountSum() > 20) {
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
            for (int i = 0; i < node.getLevel(); i++) {
                str.append("\t");
            }
            str.append(node.theta.toString()).append("\n");

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
        str.append("\t>>> # effective nodes = ").append(numEffNodes).append("\n");
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

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected final boolean extensible;
        protected SparseCount subtreeTokenCounts;
        protected SparseCount tokenCounts;
        protected double eta; // regression parameter
        protected double pi;
        protected HashMap<Integer, Double> theta;
        protected boolean newNode; // whether this node is newly created

        // estimated topics after training, which is used for test
        protected double[] phihat;

        public Node(int iter, int index, int level, boolean extendable,
                DirMult content, Node parent, double eta) {
            super(index, level, content, parent);
            this.born = iter;
            this.extensible = extendable;
            this.eta = eta;
            this.subtreeTokenCounts = new SparseCount();
            this.tokenCounts = new SparseCount();
            this.theta = new HashMap<>();
            this.newNode = true;
        }

        void setPhiHat(double[] ph) {
            this.phihat = ph;
        }

        void changeStatus() {
            this.newNode = false;
        }

        boolean isExtensible() {
            return this.extensible;
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
            // initialize
            int thetaSize = KK;
            if (isExtensible()) {
                thetaSize = KK + 1;
            }
            this.theta = new HashMap<>();
            double val = 1.0 / thetaSize;
            for (Node child : getChildren()) {
                this.theta.put(child.getIndex(), val);
            }
            if (isExtensible()) {
                this.theta.put(NEW_CHILD_INDEX, val);
            }
        }

        /**
         * Update the global theta distribution based on the current
         * approximated counts.
         */
        void updateGlobalTheta() {
            if (!isExtensible()) {
                return;
            }
            double gAlpha = getGlobalAlpha(level);

            // update counts
            SparseCount approxThetaCounts = new SparseCount();
            for (Node child : getChildren()) {
                int childIdx = child.getIndex();
                for (int dd : child.subtreeTokenCounts.getIndices()) {
                    int rawCount = child.subtreeTokenCounts.getCount(dd);
                    Double thetaVal = this.theta.get(childIdx);
                    if (thetaVal == null) { // this child has just been added
                        thetaVal = theta.get(NEW_CHILD_INDEX);
                    }
                    int approxCount = getApproxCount(rawCount, thetaVal);
                    approxThetaCounts.changeCount(childIdx, approxCount);
                }
            }

            // update theta
            this.theta = new HashMap<>();
            double norm = approxThetaCounts.getCountSum() + gAlpha;
            for (int childIdx : approxThetaCounts.getIndices()) {
                this.theta.put(childIdx,
                        (double) approxThetaCounts.getCount(childIdx) / norm);
            }
            this.theta.put(NEW_CHILD_INDEX, gAlpha / norm);
        }

        /**
         * Compute the approximated count, propagated from lower-level
         * restaurant. This can be approximated using (1) Maximal path
         * assumption, (2) Minimal path assumption, and (3) Sampling from
         * Antoniak distribution.
         *
         * @param count Actual count from lower-level restaurant
         * @param curThetaVal Current theta value
         * @return Approximate count
         */
        int getApproxCount(int count, double curThetaVal) {
            if (pathAssumption == PathAssumption.MAXIMAL) {
                return count;
            } else if (pathAssumption == PathAssumption.MINIMAL) {
                return count == 0 ? 0 : 1;
            } else if (pathAssumption == PathAssumption.ANTONIAK) {
                if (count > 1) {
                    double val = getGlobalAlpha(level) * (getNumChildren() + 1) * curThetaVal;
                    return SamplerUtils.randAntoniak(val, count);
                } else {
                    return count;
                }
            } else {
                throw new RuntimeException("Path assumption " + pathAssumption
                        + " not supported");
            }
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

        void validate(String msg) {
            this.tokenCounts.validate(msg);
            this.subtreeTokenCounts.validate(msg);

            int expectedThetaSize = getNumChildren();
            if (isExtensible()) {
                expectedThetaSize++;
            }
            if (expectedThetaSize != this.theta.size()) {
                System.out.println(this.toString());
                System.out.println(this.theta.toString());
                for (Node child : this.getChildren()) {
                    System.out.println(child.toString());
                }
                throw new RuntimeException(msg + ". Mismatch"
                        + ". " + this.theta.size()
                        + ". " + (this.getNumChildren() + 1));
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[").append(getPathString());
            str.append(", ").append(born);
            str.append(", c (").append(getChildren().size()).append(")");
            str.append(", (").append(getContent().getCountSum())
                    .append(", ").append(subtreeTokenCounts.size())
                    .append(", ").append(tokenCounts.size())
                    .append(")");
            str.append(", ").append(MiscUtils.formatDouble(eta));
            str.append(", ").append(extensible);
            str.append(", ").append(newNode);
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

    public static String hashMapToString(HashMap<Integer, Double> table) {
        if (table.isEmpty()) {
            return "";
        }
        StringBuilder str = new StringBuilder();
        for (int key : table.keySet()) {
            str.append(key).append(":").append(table.get(key)).append("\t");
        }
        return str.toString();
    }

    public static HashMap<Integer, Double> stringToHashMap(String str) {
        HashMap<Integer, Double> table = new HashMap<>();
        String[] sstr = str.split("\t");
        for (String s : sstr) {
            String[] ss = s.split(":");
            int key = Integer.parseInt(ss[0]);
            double val = Double.parseDouble(ss[1]);
            table.put(key, val);
        }
        return table;
    }

    /**
     * Run multiple test chains on test data in parallel and average to get the
     * final predictions.
     *
     * @param newDocIndices
     * @param newWords
     * @param newAuthors
     * @param newAuthorIndices
     * @param testVotes
     * @param iterPredFolder
     * @param sampler
     * @return
     */
    public static SparseVector[] parallelTest(
            ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File iterPredFolder,
            SNHDPIdealPoint sampler) {
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
                File authorScoreFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + "-"
                        + AbstractVotePredictor.AuthorScoreFile);
                File voteScoreFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + "-"
                        + AbstractVotePredictor.VoteScoreFile);
                SNHDPTestRunner runner = new SNHDPTestRunner(
                        sampler, stateFile, newDocIndices, newWords,
                        newAuthors, newAuthorIndices, testVotes,
                        partialResultFile, authorScoreFile, voteScoreFile);
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

class SNHDPTestRunner implements Runnable {

    SNHDPIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;
    File authorScoreFile;
    File voteScoreFile;

    public SNHDPTestRunner(SNHDPIdealPoint sampler,
            File stateFile,
            ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File outputFile,
            File authorScoreFile,
            File voteScoreFile) {
        this.sampler = sampler;
        this.stateFile = stateFile;
        this.testDocIndices = newDocIndices;
        this.testWords = newWords;
        this.testAuthors = newAuthors;
        this.testAuthorIndices = newAuthorIndices;
        this.testVotes = testVotes;
        this.predictionFile = outputFile;
        this.authorScoreFile = authorScoreFile;
        this.voteScoreFile = voteScoreFile;
    }

    @Override
    public void run() {
        SNHDPIdealPoint testSampler = new SNHDPIdealPoint();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());

        try {
            testSampler.test(stateFile, testDocIndices, testWords, testAuthors,
                    testAuthorIndices, testVotes,
                    predictionFile, authorScoreFile, voteScoreFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
