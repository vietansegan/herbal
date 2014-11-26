package votepredictor;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Author;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Stack;
import optimization.RidgeLinearRegressionLBFGS;
import sampler.unsupervised.RecursiveLDA;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.HTMLUtils;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.govtrack.GTLegislator;

/**
 *
 * @author vietan
 */
public class SNLDAIdealPoint extends AbstractTopicBasedIdealPoint {

    protected double[] alphas;          // [L-1]
    protected double[] betas;           // [L]
    protected double[] gamma_means;     // [L-1] mean of bias coins
    protected double[] gamma_scales;    // [L-1] scale of bias coins
    public double rho;
    public double mu;
    public double sigma;
    public int numSteps = 20;
    public double epsilon = 0.01;
    // input
    protected double[][] issuePhis;
    protected int J; // number of frames per issue
    // derive
    protected int K; // number of issues
    protected int L; // number of levels
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
    protected int numTokensChanged;
    protected double[] background;
    protected int numTokensAccepted;
    protected ArrayList<String> labelVocab;
    protected int posAnchor;
    protected int negAnchor;

    public SNLDAIdealPoint() {
        this.basename = "SNLDA-ideal-point";
    }

    public SNLDAIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setLabelVocab(ArrayList<String> labelVoc) {
        this.labelVocab = labelVoc;
    }

    public void configure(SNLDAIdealPoint sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.J,
                sampler.issuePhis,
                sampler.alphas,
                sampler.betas,
                sampler.gamma_means,
                sampler.gamma_scales,
                sampler.rho,
                sampler.mu,
                sampler.sigma,
                sampler.hasRootTopic,
                sampler.wordWeightType,
                sampler.initState,
                sampler.pathAssumption,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V, int J,
            double[][] issues,
            double[] alphas,
            double[] betas,
            double[] gamma_means,
            double[] gamma_scales,
            double rho,
            double mu, // mean of Gaussian for regression parameters
            double sigma, // stadard deviation of Gaussian for regression parameters
            boolean hasRootTopic,
            WordWeightType wwt,
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
        this.rho = rho;
        this.mu = mu;
        this.sigma = sigma;
        this.hasRootTopic = hasRootTopic;
        this.wordWeightType = wwt;

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
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg mu:\t" + MiscUtils.formatDouble(mu));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- has root topic:\t" + hasRootTopic);
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- path assumption:\t" + this.pathAssumption);
            logln("--- word weight type:\t" + this.wordWeightType);
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
        str.append("_r-").append(formatter.format(rho))
                .append("_m-").append(formatter.format(mu))
                .append("_s-").append(formatter.format(sigma));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_wwt-").append(this.wordWeightType);
        str.append("_rt-").append(this.hasRootTopic);
        str.append("_").append(pathAssumption);
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

    protected double getAlpha(int l) {
        return this.alphas[l];
    }

    protected double getBeta(int l) {
        return this.betas[l];
    }

    protected double getGammaMean(int l) {
        return this.gamma_means[l];
    }

    protected double getGammaScale(int l) {
        return this.gamma_scales[l];
    }

    public double[] getPredictedUs() {
        return this.authorMeans;
    }

    @Override
    public String getCurrentState() {
        return this.getSamplerFolderPath()
                + "\n" + printGlobalTreeSummary()
                + "\nCurrent thread " + Thread.currentThread().getId();
    }

    /**
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @return Predicted probabilities
     */
    public SparseVector[] predictInMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    /**
     * Make predictions on held-out votes of unknown legislators on known votes.
     *
     * @return Predicted probabilities
     */
    public SparseVector[] predictOutMatrix() {
        this.u = this.authorMeans;
        return predictInMatrix();
    }

    /**
     * Sample topic assignments for test documents and make predictions.
     *
     * @param stateFile
     * @param predictionFile
     * @param assignmentFile
     * @return predictions
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
        setTestConfigurations(); // set up test

        if (verbose) {
            logln("Setting up test ...");
            logln("--- test burn-in: " + testBurnIn);
            logln("--- test maximum number of iterations: " + testMaxIter);
            logln("--- test sample lag: " + testSampleLag);
            logln("--- test report interval: " + testRepInterval);
        }

        // sample on test data
        sampleNewDocuments(stateFile, assignmentFile);

        // make prediction on votes of unknown voters
        SparseVector[] predictions = predictOutMatrix();

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
                        + ". # tokens changed: " + numTokensChanged
                        + " (" + (double) numTokensChanged / numTokens + ")"
                        + ". # tokens accepted: " + numTokensAccepted
                        + " (" + (double) numTokensAccepted / numTokens + ")"
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

    @Override
    public void initialize() {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        initializeModelStructure();
        initializeDataStructure();
        initializeUXY();
        initializeAssignments();
        updateEtas();
        if (debug) {
            validate("Initialized");
        }
        if (verbose) {
            logln("--- Done initializing. \t" + getCurrentState());
            getLogLikelihood();
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
        this.root = new Node(iter, 0, 0, rootTopic, null, 0.0);
        for (int kk = 0; kk < K; kk++) {
            DirMult issueTopic = new DirMult(V, getBeta(1) * V,
                    rlda.getTopicWord(new int[]{kk}).getDistribution());
            Node issueNode = new Node(iter, kk, 1, issueTopic, root,
                    SamplerUtils.getGaussian(mu, sigma));
            root.addChild(kk, issueNode);
            for (int jj = 0; jj < J; jj++) {
                DirMult frameTopic = new DirMult(V, getBeta(2) * V,
                        rlda.getTopicWord(new int[]{kk, jj}).getDistribution());
                Node frameNode = new Node(iter, jj, 2, frameTopic, issueNode,
                        SamplerUtils.getGaussian(mu, sigma));
                issueNode.addChild(jj, frameNode);
            }
        }
        this.root.initializeGlobalTheta();
        this.root.initializeGlobalPi();
        for (Node issueNode : this.root.getChildren()) {
            issueNode.initializeGlobalTheta();
            issueNode.initializeGlobalPi();
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
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
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
                String str = "\n\nIter " + iter + "/" + MAX_ITER
                        + "\t llh = " + loglikelihood
                        + "\n" + getCurrentState();
                if (iter < BURN_IN) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            long etaTime = updateEtas();
            long uxyTime = updateUXY();
            long topicTime = sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            if (isReporting) {
                logln("\n" + printGlobalTree() + "\n");
                logln("--- --- Time. Topic: " + topicTime
                        + ". Eta: " + etaTime
                        + ". UXY: " + uxyTime);
                logln("--- --- # tokens: " + numTokens
                        + ". # tokens changed: " + numTokensChanged
                        + " (" + (double) numTokensChanged / numTokens + ")"
                        + ". # tokens accepted: " + numTokensAccepted
                        + " (" + (double) numTokensAccepted / numTokens + ")");
                logln("--- --- positive anchor (" + posAnchor + "): "
                        + MiscUtils.formatDouble(u[posAnchor])
                        + ". negative anchor (" + negAnchor + "): "
                        + MiscUtils.formatDouble(u[negAnchor]));
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
            authorMeans[authors[dd]] += node.eta * wordWeights[words[dd][nn]]
                    / authorTotalWordWeights[authors[dd]];
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
            if (authorMeans != null) {
                authorMeans[authors[dd]] -= node.eta * wordWeights[words[dd][nn]]
                        / authorTotalWordWeights[authors[dd]];
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
        long sTime = System.currentTimeMillis();
        numTokensChanged = 0;
        numTokensAccepted = 0;
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                // remove
                removeToken(dd, nn, z[dd][nn], removeFromData, removeFromModel);

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
            logprobs[ACTUAL_INDEX] += getResponseLogLikelihood(authors[dd], words[dd][nn], node);
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
     * @param author The author
     * @param node The node
     * @return
     */
    private double getResponseLogLikelihood(int author, int word, Node node) {
        double aMean = authorMeans[author] + node.eta * wordWeights[word]
                / authorTotalWordWeights[author];
        double resLLh = StatUtils.logNormalProbability(u[author], aMean, Math.sqrt(rho));
        return resLLh;
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

        // list of nodes
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        // design matrix
        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(N);
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                int kk = nodeList.indexOf(z[dd][nn]);
                designMatrix[aa].change(kk, wordWeights[words[dd][nn]]
                        / authorTotalWordWeights[aa]);
            }
        }

        // current params
        double[] etas = new double[N];
        for (int kk = 0; kk < N; kk++) {
            etas[kk] = nodeList.get(kk).eta;
        }

        RidgeLinearRegressionLBFGS optimizable = new RidgeLinearRegressionLBFGS(
                u, etas, designMatrix, rho, mu, sigma);

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
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
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

    /**
     * Update U, X and Y.
     *
     * @return Elapsed time
     */
    private long updateUXY() {
        if (isReporting) {
            logln("+++ Updating UXY ...");
        }
        long sTime = System.currentTimeMillis();
        for (int ii = 0; ii < numSteps; ii++) {
            updateUs();
            updateXYs();
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    /**
     * Update Us.
     */
    private void updateUs() {
        for (int a = 0; a < A; a++) {
            double grad = 0.0;
            // likelihood
            for (int b = 0; b < B; b++) {
                if (isValidVote(a, b)) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    double prob = score / (1 + score);
                    grad += x[b] * (getVote(a, b) - prob); // only work for 0 and 1
                }
            }
            // prior
            grad -= (u[a] - authorMeans[a]) / sigma;
            // update
            u[a] += epsilon * grad;
        }
    }

    /**
     * Update Xs and Ys.
     */
    public void updateXYs() {
        for (int b = 0; b < B; b++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int a = 0; a < A; a++) {
                if (isValidVote(a, b)) {
                    double score = Math.exp(u[a] * x[b] + y[b]);
                    gradX += u[a] * (getVote(a, b) - score / (1 + score));
                    gradY += getVote(a, b) - score / (1 + score);
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
        for (int ii = 0; ii < A; ii++) {
            for (int jj = 0; jj < B; jj++) {
                if (isValidVote(ii, jj)) {
                    double score = u[ii] * x[jj] + y[jj];
                    voteLlh += getVote(ii, jj) * score - Math.log(1 + Math.exp(score));
                }
            }
        }
        return voteLlh;
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
        logln(msg + ". Validation not implemented!");
    }

    @Override
    public void outputState(String filepath) {
        if (verbose) {
            logln("--- Outputing current state to " + filepath);
        }

        // authors
        StringBuilder authorStr = new StringBuilder();
        for (int aa = 0; aa < A; aa++) {
            authorStr.append(aa).append("\t")
                    .append(this.authorIndices.get(aa)).append("\t")
                    .append(u[aa]).append("\n");
        }

        // bills
        StringBuilder billStr = new StringBuilder();
        for (int bb = 0; bb < B; bb++) {
            billStr.append(bb).append("\t")
                    .append(this.billIndices.get(bb)).append("\t")
                    .append(x[bb]).append("\t").append(y[bb]).append("\n");
        }

        // model string
        StringBuilder modelStr = new StringBuilder();
        for (int vv = 0; vv < V; vv++) {
            modelStr.append(vv).append("\t").append(this.wordWeights[vv]).append("\n");
        }
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

        try { // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());
            contentStrs.add(authorStr.toString());
            contentStrs.add(billStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + AuthorFileExt);
            entryFiles.add(filename + BillFileExt);

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
            inputAuthorScore(filepath);
            inputBillScore(filepath);
            inputAssignments(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + filepath);
        }
    }

    public void inputAuthorScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author score from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AuthorFileExt);

            u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != aa) {
                    throw new MismatchRuntimeException(Integer.parseInt(sline[0]), aa);
                }
                if (Integer.parseInt(sline[1]) != authorIndices.get(aa)) {
                    throw new MismatchRuntimeException(Integer.parseInt(sline[1]),
                            authorIndices.get(aa));
                }
                u[aa] = Double.parseDouble(sline[2]);
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
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + BillFileExt);
            x = new double[B];
            y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != bb) {
                    throw new MismatchRuntimeException(Integer.parseInt(sline[0]), bb);
                }
                if (Integer.parseInt(sline[1]) != billIndices.get(bb)) {
                    throw new MismatchRuntimeException(Integer.parseInt(sline[1]),
                            billIndices.get(bb));
                }
                x[bb] = Double.parseDouble(sline[2]);
                y[bb] = Double.parseDouble(sline[3]);
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
            wordWeights = new double[V];
            for (int vv = 0; vv < V; vv++) {
                String[] sline = reader.readLine().split("\t");
                int vIdx = Integer.parseInt(sline[0]);
                if (vv != vIdx) {
                    throw new MismatchRuntimeException(vIdx, vv);
                }
                wordWeights[vv] = Double.parseDouble(sline[1]);
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

                Node node = new Node(born, nodeIndex, nodeLevel, topic, parent, eta);
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
            reader.close();
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

    /**
     * Get the features extracted.
     *
     * @return
     */
    public SparseVector[] getAuthorFeatures() {
        ArrayList<Node> nodeList = getNodeList();
        int N = nodeList.size();

        SparseVector[] authorNodeProps = new SparseVector[A];
        SparseVector[] authorNodeWeightedProps = new SparseVector[A];
        SparseVector[] authorNodePolarizedProps = new SparseVector[A];

        for (int aa = 0; aa < A; aa++) {
            authorNodeProps[aa] = new SparseVector(N);
            authorNodeWeightedProps[aa] = new SparseVector(N);
            authorNodePolarizedProps[aa] = new SparseVector(N);
        }

        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                int kk = nodeList.indexOf(z[dd][nn]);
                authorNodeProps[aa].change(kk, 1.0);
                authorNodeWeightedProps[aa].change(kk, wordWeights[words[dd][nn]]);
                authorNodePolarizedProps[aa].change(kk, wordWeights[words[dd][nn]]
                        * z[dd][nn].eta);
            }
        }

        SparseVector[] authorFeatures = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            authorFeatures[aa] = new SparseVector(3 * N + 1);
        }
        for (int aa = 0; aa < A; aa++) {
            authorNodeProps[aa].normalize();
            for (int kk = 0; kk < N; kk++) {
                authorFeatures[aa].change(kk, authorNodeProps[aa].get(kk));
            }

            authorNodeWeightedProps[aa].normalize();
            for (int kk = 0; kk < N; kk++) {
                authorFeatures[aa].change(N + kk, authorNodeWeightedProps[aa].get(kk));
            }

            authorNodePolarizedProps[aa].normalize();
            for (int kk = 0; kk < N; kk++) {
                authorFeatures[aa].change(2 * N + kk, authorNodePolarizedProps[aa].get(kk));
            }

            authorFeatures[aa].change(3 * N, authorNodePolarizedProps[aa].sum());
        }
        return authorFeatures;
    }

    /**
     * Get regression parameters of first-level topics.
     *
     * @return
     */
    public double[] getTopicScores() {
        double[] topicScores = new double[K];
        for (int kk = 0; kk < K; kk++) {
            topicScores[kk] = this.root.getChild(kk).eta;
        }
        return topicScores;
    }

    public void outputHTML(File htmlFile,
            ArrayList<Integer> docIndices,
            String[] docIds,
            String[][] docSentRawTexts,
            HashMap<String, Author> authorTable,
            String govtrackUrl) {
        if (verbose) {
            logln("--- Outputing result to HTML file " + htmlFile);
        }

        StringBuilder str = new StringBuilder();
        str.append("<table>\n");
        str.append("<tbody>\n");

        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            ArrayList<RankingItem<Node>> rankChildren = new ArrayList<RankingItem<Node>>();
            for (Node child : node.getChildren()) {
                rankChildren.add(new RankingItem<Node>(child, child.eta));
            }
            Collections.sort(rankChildren);
            for (RankingItem<Node> rankChild : rankChildren) {
                stack.add(rankChild.getObject());
            }

            if (node.getLevel() == 1) {
                String color = HTMLUtils.getColor(node.eta);
                str.append("<tr class=\"level2\"><td style=\"background-color:#FFFFFF\"></td></tr>\n");
                str.append("<tr class=\"level").append(node.getLevel()).append("\">\n");
                str.append("<td style=\"background-color:").append(color)
                        .append(";color:").append(HTMLUtils.getTextColor(color))
                        .append(";\"").append(">\n")
                        .append("<a style=\"text-decoration:underline;color:blue;\" onclick=\"showme('")
                        .append(node.getPathString())
                        .append("');\" id=\"toggleDisplay\">")
                        .append("[")
                        .append(labelVocab.get(node.getIndex()))
                        .append("]</a>")
                        .append(" (").append(node.getContent().getCountSum())
                        .append(", ").append(MiscUtils.formatDouble(node.eta))
                        .append(")");
                String[] topWords = node.getTopWords(15);
                for (String word : topWords) {
                    str.append(" ").append(word);
                }
                str.append("</td>\n");
                str.append("</tr>\n");
            } else if (node.getLevel() == 2) {
                String color = HTMLUtils.getColor(node.eta);
                str.append("<tr class=\"level2\"><td style=\"background-color:#FFFFFF\"></td></tr>\n");
                str.append("<tr class=\"level").append(node.getLevel()).append("\">\n");
                str.append("<td style=\"background-color:").append(color)
                        .append(";color:").append(HTMLUtils.getTextColor(color))
                        .append(";\"").append(">\n")
                        .append("<a style=\"text-decoration:underline;color:blue;\" onclick=\"showme('")
                        .append(node.getPathString())
                        .append("');\" id=\"toggleDisplay\">")
                        .append("[")
                        .append(node.getPathString())
                        .append("]</a>")
                        .append(" (").append(node.getContent().getCountSum())
                        .append(", ").append(MiscUtils.formatDouble(node.eta))
                        .append(")");
                String[] topWords = node.getTopWords(15);
                for (String word : topWords) {
                    str.append(" ").append(word);
                }
                str.append("</td>\n");
                str.append("</tr>\n");

                // snippets
                ArrayList<RankingItem<Integer>> rankDocs = rankDocuments(node);
                str.append("<tr class=\"level").append(node.getLevel()).append("\"")
                        .append(" id=\"").append(node.getPathString()).append("\"")
                        .append(" style=\"display:none;\"")
                        .append(">\n");
                str.append("<td>\n");
                str.append("<ol>\n");
                for (int i = 0; i < Math.min(10, rankDocs.size()); i++) {
                    RankingItem<Integer> rankDoc = rankDocs.get(i);
                    int ii = rankDoc.getObject();
                    int dd = docIndices.get(ii);

                    // doc id
                    String debateId = docIds[dd].split("_")[0];
                    str.append("<li>[")
                            .append(HTMLUtils.getEmbeddedLink(debateId,
                                            govtrackUrl + debateId + ".xml"))
                            .append("] ");
                    str.append("</li>\n");

                    ArrayList<String> docInfoList = new ArrayList<>();

                    // doc text
                    StringBuilder docStr = new StringBuilder();
                    for (String ss : docSentRawTexts[dd]) {
                        docStr.append(ss).append(" ");
                    }
                    docInfoList.add(docStr.toString());

                    // author
                    String authorId = authorVocab.get(authorIndices.get(authors[ii]));
                    String authorName = authorTable.get(authorId).getProperty(GTLegislator.NAME);
                    String authorFWID = authorTable.get(authorId).getProperty(GTLegislator.FW_ID);
                    docInfoList.add(HTMLUtils.getEmbeddedLink(authorName,
                            "http://congress.freedomworks.org/node/" + authorFWID));

                    String authorParty = authorTable.get(authorId).getProperty(GTLegislator.PARTY);
                    docInfoList.add(authorParty);

                    String nominateScore = authorTable.get(authorId).getProperty(GTLegislator.NOMINATE_SCORE1);
                    docInfoList.add("DW-NOMINATE Score: " + nominateScore);

//                    String fwScore = authorTable.get(authorId).getProperty(GTLegislator.FW_SCORE);
//                    docInfoList.add("FW Score: " + fwScore);
//                    docInfoList.add("Estmated score: " + u[authors[ii]]);
                    str.append(HTMLUtils.getHTMLList(docInfoList)).append("\n");
                }
                str.append("</ol>\n");
                str.append("</td>\n");
                str.append("</tr>\n");
            }
        }
        str.append("</tbody>\n");
        str.append("</table>\n");
        HTMLUtils.outputHTMLFile(htmlFile, str.toString());
    }

    private ArrayList<RankingItem<Integer>> rankDocuments(Node node) {
        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<>();
        for (int ii : node.tokenCounts.getIndices()) {
            if (words[ii].length < 20) {
                continue;
            }
            double val = (double) node.tokenCounts.getCount(ii) / words[ii].length;
            rankDocs.add(new RankingItem<Integer>(ii, val));
        }
        Collections.sort(rankDocs);
        return rankDocs;
    }

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected SparseCount subtreeTokenCounts;
        protected SparseCount tokenCounts;
        protected double eta; // regression parameter
        protected double pi;
        protected double[] theta;

        // estimated topics after training, which is used for test
        protected double[] phihat;

        public Node(int iter, int index, int level, DirMult content, Node parent,
                double eta) {
            super(index, level, content, parent);
            this.born = iter;
            this.eta = eta;
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
            str.append(", c (").append(getChildren().size()).append(")");
            // word types
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
            SNLDAIdealPoint sampler) {
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
                File assignmentFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + ".zip");
                File authorScoreFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + "-"
                        + AbstractVotePredictor.AuthorScoreFile);
                File voteScoreFile = new File(iterPredFolder,
                        IOUtils.removeExtension(filename) + "-"
                        + AbstractVotePredictor.VoteScoreFile);
                SNLDATestRunner runner = new SNLDATestRunner(
                        sampler, stateFile, newDocIndices, newWords,
                        newAuthors, newAuthorIndices, testVotes,
                        partialResultFile, authorScoreFile, voteScoreFile,
                        assignmentFile);
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

class SNLDATestRunner implements Runnable {

    SNLDAIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;
    File authorScoreFile;
    File voteScoreFile;
    File assignmentFile;

    public SNLDATestRunner(SNLDAIdealPoint sampler,
            File stateFile,
            ArrayList<Integer> newDocIndices,
            int[][] newWords,
            int[] newAuthors,
            ArrayList<Integer> newAuthorIndices,
            boolean[][] testVotes,
            File outputFile,
            File authorScoreFile,
            File voteScoreFile,
            File assignmentFile) {
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
        this.assignmentFile = assignmentFile;
    }

    @Override
    public void run() {
        SNLDAIdealPoint testSampler = new SNLDAIdealPoint();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());
        testSampler.setupData(testDocIndices, testWords, testAuthors, null,
                testAuthorIndices, null, testVotes);
        try {
            testSampler.test(stateFile, predictionFile, assignmentFile);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
