package votepredictor.textidealpoint;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Author;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import optimization.OWLQNLinearRegression;
import optimization.RidgeLinearRegressionOptimizable;
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.RankingItem;
import util.SparseVector;
import sampling.likelihood.CascadeDirMult.PathAssumption;
import util.MismatchRuntimeException;
import util.SamplerUtils;
import util.StatUtils;
import util.evaluation.Measurement;
import util.govtrack.GTLegislator;
import votepredictor.AbstractVotePredictor;

/**
 *
 * @author vietan
 */
public class HybridSNHDPIdealPoint extends AbstractTextSingleIdealPoint {

    public static final int PROPOSE_INDEX = 0;
    public static final int ASSIGN_INDEX = 1;
    public static final int INVALID_ISSUE_INDEX = -1;
    public static final int NEW_CHILD_INDEX = -1;
    // hyperparameters for fixed-height tree
    protected double[] globalAlphas;   // [L-1]
    protected double[] localAlphas;    // [L-1]
    protected double[] betas;           // [L]
    protected double[] sigmas;          // [L-1] eta variance (excluding root)
    protected double pi;                // mean of bias coins
    protected double[] gammas;          // [L-1] scale of bias coins
    protected double lambda;
    protected double l1;
    protected double l2;
    // input
    protected int K; // number of issues
    protected int L; // number of levels
    // config
    protected boolean hasRootTopic;
    protected PathAssumption pathAssumption;
    // latent
    Node root;
    Node[][] z;
    protected SparseVector[] tau; // each topic has its own lexical regression
    protected double[] topicVals;
    protected double[] lexicalVals;
    // internal
    protected ArrayList<String> labelVocab;
    protected int numTokensAccepted;
    protected double uniform;

    public HybridSNHDPIdealPoint() {
        this.basename = "Hybrid-SNHDP-ideal-point";
    }

    public HybridSNHDPIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setLabelVocab(ArrayList<String> labVoc) {
        this.labelVocab = labVoc;
    }

    public void configure(HybridSNHDPIdealPoint sampler) {
        if (sampler.lambda > 0) {
            this.configure(sampler.folder,
                    sampler.V,
                    sampler.K,
                    sampler.globalAlphas,
                    sampler.localAlphas,
                    sampler.betas,
                    sampler.pi,
                    sampler.gammas,
                    sampler.rho,
                    sampler.sigmas,
                    sampler.sigma,
                    sampler.lambda,
                    sampler.hasRootTopic,
                    sampler.initState,
                    sampler.pathAssumption,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        } else {
            this.configure(sampler.folder,
                    sampler.V,
                    sampler.K,
                    sampler.globalAlphas,
                    sampler.localAlphas,
                    sampler.betas,
                    sampler.pi,
                    sampler.gammas,
                    sampler.rho,
                    sampler.sigmas,
                    sampler.sigma,
                    sampler.l1,
                    sampler.l2,
                    sampler.hasRootTopic,
                    sampler.initState,
                    sampler.pathAssumption,
                    sampler.paramOptimized,
                    sampler.BURN_IN,
                    sampler.MAX_ITER,
                    sampler.LAG,
                    sampler.REP_INTERVAL);
        }
    }

    public void configure(String folder,
            int V, int K,
            double[] global_alphas,
            double[] local_alphas,
            double[] betas,
            double pi,
            double[] gammas,
            double rho,
            double[] sigmas, // stadard deviation of Gaussian for regression parameters
            double ipSigma,
            double lambda,
            boolean hasRootTopic,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.K = K;
        this.L = betas.length;
        this.uniform = 1.0 / V;

        this.globalAlphas = global_alphas;
        this.localAlphas = local_alphas;
        this.betas = betas;
        this.pi = pi;
        this.gammas = gammas;
        this.rho = rho;
        this.sigmas = sigmas;
        this.sigma = ipSigma;
        this.lambda = lambda;
        this.hasRootTopic = hasRootTopic;
        this.wordWeightType = WordWeightType.NONE;

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

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K);
        str.append("_ga");
        for (double ga : globalAlphas) {
            str.append("-").append(MiscUtils.formatDouble(ga));
        }
        str.append("_la");
        for (double la : localAlphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_p-").append(formatter.format(pi));
        str.append("_g");
        for (double g : gammas) {
            str.append("-").append(MiscUtils.formatDouble(g));
        }
        str.append("_s");
        for (double s : sigmas) {
            str.append("-").append(MiscUtils.formatDouble(s));
        }
        str.append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma))
                .append("_l-").append(formatter.format(lambda));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_rt-").append(this.hasRootTopic);
        this.name = str.toString();

        if (verbose) {
            logln("--- V = " + this.V);
            logln("--- K = " + this.K);
            logln("--- L = " + this.L);
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(local_alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- pi:\t" + MiscUtils.formatDouble(pi));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- ideal point sigma:\t" + MiscUtils.formatDouble(ipSigma));
            logln("--- has root topic:\t" + hasRootTopic);
            logln("--- lambda:\t" + MiscUtils.formatDouble(this.lambda));
            logln("--- l1:\t" + MiscUtils.formatDouble(this.l1, 10));
            logln("--- l2:\t" + MiscUtils.formatDouble(this.l2, 10));
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

    public void configure(String folder,
            int V, int K,
            double[] global_alphas,
            double[] local_alphas,
            double[] betas,
            double pi,
            double[] gammas,
            double rho,
            double[] sigmas, // stadard deviation of Gaussian for regression parameters
            double ipSigma,
            double l1, double l2,
            boolean hasRootTopic,
            InitialState initState,
            PathAssumption pathAssumption,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.K = K;
        this.L = betas.length;
        this.uniform = 1.0 / V;

        this.globalAlphas = global_alphas;
        this.localAlphas = local_alphas;
        this.betas = betas;
        this.pi = pi;
        this.gammas = gammas;
        this.rho = rho;
        this.sigmas = sigmas;
        this.sigma = ipSigma;
        this.l1 = l1;
        this.l2 = l2;
        this.hasRootTopic = hasRootTopic;
        this.wordWeightType = WordWeightType.NONE;

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

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K);
        str.append("_ga");
        for (double ga : globalAlphas) {
            str.append("-").append(MiscUtils.formatDouble(ga));
        }
        str.append("_la");
        for (double la : localAlphas) {
            str.append("-").append(MiscUtils.formatDouble(la));
        }
        str.append("_b");
        for (double b : betas) {
            str.append("-").append(MiscUtils.formatDouble(b));
        }
        str.append("_p-").append(formatter.format(pi));
        str.append("_g");
        for (double g : gammas) {
            str.append("-").append(MiscUtils.formatDouble(g));
        }
        str.append("_s");
        for (double s : sigmas) {
            str.append("-").append(MiscUtils.formatDouble(s));
        }
        str.append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma))
                .append("_l1-").append(MiscUtils.formatDouble(l1, 10))
                .append("_l2-").append(MiscUtils.formatDouble(l2, 10));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_rt-").append(this.hasRootTopic);
        this.name = str.toString();

        if (verbose) {
            logln("--- V = " + this.V);
            logln("--- K = " + this.K);
            logln("--- L = " + this.L);
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(local_alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- pi:\t" + MiscUtils.formatDouble(pi));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- ideal point sigma:\t" + MiscUtils.formatDouble(ipSigma));
            logln("--- lambda:\t" + MiscUtils.formatDouble(this.lambda));
            logln("--- l1:\t" + MiscUtils.formatDouble(this.l1, 10));
            logln("--- l2:\t" + MiscUtils.formatDouble(this.l2, 10));
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

    protected double getLocalAlpha(int l) {
        return this.localAlphas[l];
    }

    protected double getGlobalAlpha(int l) {
        return this.globalAlphas[l];
    }

    protected double getBeta(int l) {
        return this.betas[l];
    }

    protected double getPi() {
        return this.pi;
    }

    protected double getGamma(int l) {
        return this.gammas[l];
    }

    protected double getSigma(int l) {
        return this.sigmas[l - 1];
    }

    public double[] getPredictedUs() {
        double[] predUs = new double[A];
        for (int aa = 0; aa < A; aa++) {
            predUs[aa] = topicVals[aa] + lexicalVals[aa];
        }
        return predUs;
    }

    @Override
    public void initialize() {
        initialize(null);
    }

    public void initialize(double[][] issuePhis) {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;

        initializeIdealPoint(); // initialize ideal points
        initializeModelStructure(issuePhis);
        initializeDataStructure();
        initializeAssignments();
        if (this.lambda > 0) {
            updateTausLBFGS();
        } else {
            updateTausOWLQN();
        }
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
    protected void initializeModelStructure(double[][] issuePhis) {
        if (issuePhis != null && issuePhis.length != K) {
            throw new MismatchRuntimeException(issuePhis.length, K);
        }
        LDA lda = runLDA(words, K, V, issuePhis);

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, 1.0 / V);
        this.root = new Node(iter, 0, 0, rootTopic, null, INVALID_ISSUE_INDEX, 0.0, false);
        this.root.changeStatus();
        for (int kk = 0; kk < K; kk++) {
            DirMult issueTopic = new DirMult(V, getBeta(1) * V,
                    lda.getTopicWords()[kk].getDistribution());
            Node issueNode = new Node(iter, kk, 1, issueTopic, root, kk,
                    SamplerUtils.getGaussian(0.0, getSigma(1)), true);
            issueNode.changeStatus();
            root.addChild(kk, issueNode);
        }

        // initialize global theta at root
        double thetaUnif = 1.0 / K;
        for (int kk = 0; kk < K; kk++) {
            this.root.theta.put(kk, thetaUnif);
        }

        tau = new SparseVector[K];
        for (int kk = 0; kk < K; kk++) {
            tau[kk] = new SparseVector(V);
        }

        if (verbose && debug) {
            logln("Initialized.\n" + printGlobalTree());
        }
    }

    protected void initializeDataStructure() {
        z = new Node[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new Node[words[d].length];
        }
        topicVals = new double[A];
        lexicalVals = new double[A];
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
        for (int dd = 0; dd < D; dd++) {
            for (int nn = 0; nn < words[dd].length; nn++) {
                Node sampledNode = sampleNode(dd, nn, root, EXTEND);
                z[dd][nn] = sampledNode;
                addToken(dd, nn, z[dd][nn], ADD, ADD);

                if (z[dd][nn].isNew) {
                    z[dd][nn].changeStatus();
                    z[dd][nn].getParent().updateTheta();
                }
            }
        }
    }

    /**
     * Make predictions on held-out votes of unknown legislators on known votes.
     *
     * @return Predicted probabilities
     */
    @Override
    public SparseVector[] predictOutMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double score = Math.exp((topicVals[aa] + lexicalVals[aa]) * x[bb] + y[bb]);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    protected double getMSE() {
        double mse = 0.0;
        for (int aa = 0; aa < A; aa++) {
            double diff = u[aa] - (topicVals[aa] + lexicalVals[aa]);
            mse += diff * diff;
        }
        return mse / A;
    }

    @Override
    public String getCurrentState() {
        if (debug) {
            System.out.println(printGlobalTree());
        }
        String str = "\n\nIter " + iter + "/" + MAX_ITER
                + "\t @ " + Thread.currentThread().getId();
        if (iter < BURN_IN) {
            logln("--- Burning in. " + str);
        } else {
            logln("--- Sampling. " + str);
        }
        logln("--- Evaluating ...");
        SparseVector[] predictions = predictInMatrix();
        ArrayList<Measurement> measurements = AbstractVotePredictor
                .evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> i >>> " + m.getName() + ": " + m.getValue());
        }

        predictions = predictOutMatrix();
        measurements = AbstractVotePredictor
                .evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> o >>> " + m.getName() + ": " + m.getValue());
        }
        logln("--- MSE: " + getMSE());
        return str;
    }

    @Override
    public void iterate() {
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED, EXTEND);
        updateThetasAndPis();

        if (this.lambda > 0) {
            updateTausLBFGS();
        } else {
            updateTausOWLQN();
        }
        updateEtas();

        if (iter >= BURN_IN && iter % LAG == 0) {
            updateUXY();
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
            topicVals[authors[dd]] += node.pathEta / authorTotalWordWeights[authors[dd]];
            lexicalVals[authors[dd]] += tau[node.issueIndex].get(words[dd][nn])
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
            topicVals[authors[dd]] -= node.pathEta / authorTotalWordWeights[authors[dd]];
            lexicalVals[authors[dd]] -= tau[node.issueIndex].get(words[dd][nn])
                    / authorTotalWordWeights[authors[dd]];
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
                    parent.updateTheta();
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
        if (isReporting) {
            logln("+++ Sampling Zs ...");
        }
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
                        if (sampledNode.isNew) {
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
                    if (z[dd][nn].isNew) {
                        z[dd][nn].changeStatus();
                        parent.updateTheta();
                    }
                } else {
                    // if reject the proposed node and the current node is removed
                    // from the tree, we need to add it back to the tree
                    if (!z[dd][nn].isRoot() && !parent.hasChild(zIdx)) {
                        parent.addChild(zIdx, z[dd][nn]);
                        parent.updateTheta();
                    }
                }
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
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
        if (curNode.isLeaf() && !curNode.extensible) {
            return curNode;
        }

        int level = curNode.getLevel();
        double gamma = getGamma(level);
        double stayprob = 0.0;
        if (hasRootTopic || (!hasRootTopic && !curNode.isRoot())) {
            stayprob = (curNode.tokenCounts.getCount(dd) + gamma * curNode.pi)
                    / (curNode.subtreeTokenCounts.getCount(dd) + gamma);
        }
        double passprob = 1.0 - stayprob;

        ArrayList<Node> nodeList = new ArrayList<>();
        ArrayList<Double> nodeProbs = new ArrayList<>();

        // for staying at current node
        nodeList.add(curNode);
        nodeProbs.add(stayprob * curNode.getPhi(words[dd][nn]));

        // for moving to an existing child node
        double lAlpha = getLocalAlpha(level);
        double norm = curNode.getPassingCount(dd) + lAlpha;
        for (Node child : curNode.getChildren()) {
            int childIdx = child.getIndex();
            nodeList.add(child);

            if (curNode.theta.get(childIdx) == null) {
                System.out.println("iter = " + iter);
                System.out.println("curnode: " + curNode.toString());
                System.out.println("child: " + child.toString());
                throw new RuntimeException();
            }

            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * curNode.theta.get(childIdx)) / norm;
            nodeProbs.add(passprob * pathprob * child.getPhi(words[dd][nn]));
        }

        // new child node
        if (extend && curNode.extensible) {
            nodeList.add(null);
            double pathprob = lAlpha * curNode.theta.get(NEW_CHILD_INDEX) / norm;
            nodeProbs.add(passprob * pathprob * uniform);
        }

        int sampledIdx = SamplerUtils.scaleSample(nodeProbs);
        Node sampledNode = nodeList.get(sampledIdx);
        if (sampledNode == null) { // new child
            int nodeIdx = curNode.getNextChildIndex();
            int nodeLevel = curNode.getLevel() + 1;
            boolean extendable = nodeLevel != L - 1;
            DirMult topic = new DirMult(V, getBeta(nodeLevel) * V, uniform);
            double eta = SamplerUtils.getGaussian(0.0, getSigma(level + 1));
            Node newNode = new Node(iter, nodeIdx, nodeLevel, topic, curNode, curNode.issueIndex,
                    eta, extendable);
            curNode.addChild(nodeIdx, newNode);

            // update theta value for this new child node
            double val = getGlobalAlpha(curNode.getLevel())
                    / (curNode.subtreeTokenCounts.getCountSum()
                    - curNode.tokenCounts.getCountSum() + 1); // TODO: check this
            curNode.theta.put(nodeIdx, val);
            return newNode;
        } else if (sampledNode.equals(curNode)) { // stay at current node
            return sampledNode;
        } else { // recursively move to an existing child
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
            int aa = authors[dd];
            double aMean = topicVals[aa] + lexicalVals[aa]
                    + (tau[node.issueIndex].get(words[dd][nn]) + node.pathEta)
                    / authorTotalWordWeights[aa];
            logprobs[ACTUAL_INDEX] += StatUtils.logNormalProbability(u[aa], aMean, Math.sqrt(rho));
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
        double gamma = getGamma(level);
        double stayprob = (source.tokenCounts.getCount(dd) + gamma * source.pi)
                / (source.subtreeTokenCounts.getCount(dd) + gamma);
        double passprob = 1.0 - stayprob;
        double norm = source.subtreeTokenCounts.getCount(dd)
                - source.tokenCounts.getCount(dd) + lAlpha;

        // existing children
        boolean foundTarget = false;
        for (Node child : source.getChildren()) {
            double pathprob;
            double wordprob;
            if (child.isNew) { // newly created child
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
        double[] etaVars = new double[N];
        double[] etas = new double[N];
        for (int kk = 0; kk < N; kk++) {
            Node node = nodeList.get(kk);
            etas[kk] = node.eta;
            etaVars[kk] = getSigma(node.getLevel());

            for (int dd : node.tokenCounts.getIndices()) {
                int count = node.tokenCounts.getCount(dd);
                int author = authors[dd];
                double val = (double) count / authorTotalWordWeights[author];

                Node tempNode = node;
                while (!tempNode.isRoot()) {
                    int nodeIdx = nodeList.indexOf(tempNode);
                    if (nodeIdx < 0) {
                        throw new RuntimeException("Index: " + nodeIdx
                                + ". Node: " + tempNode);
                    }
                    designMatrix[author].change(nodeIdx, val);
                    tempNode = tempNode.getParent();
                }
            }
        }

        double[] responses = new double[A];
        for (int aa = 0; aa < A; aa++) {
            responses[aa] = u[aa] - lexicalVals[aa];
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, etas, designMatrix, rho, 0.0, etaVars);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update regression parameters
        for (int kk = 0; kk < N; kk++) {
            nodeList.get(kk).eta = optimizable.getParameter(kk);
        }
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            for (Node child : node.getChildren()) {
                queue.add(child);
            }
            node.updatePathEta();
        }

        // update topical regression values
        for (int aa = 0; aa < A; aa++) {
            topicVals[aa] = 0.0;
            for (int kk : designMatrix[aa].getIndices()) {
                topicVals[aa] += designMatrix[aa].get(kk) * nodeList.get(kk).eta;
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
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
     * Update the global theta and pi.
     *
     * @return Elapsed time
     */
    private long updateThetasAndPis() {
        if (isReporting) {
            logln("+++ Updating theta's and pi's ...");
        }
        long sTime = System.currentTimeMillis();
        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            for (Node child : node.getChildren()) {
                stack.add(child);
            }
            node.updateTheta();
            node.updatePi();
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    protected long updateTausLBFGS() {
        if (isReporting) {
            logln("+++ Updating tau using L-BFGS ...");
        }
        long sTime = System.currentTimeMillis();

        SparseCount[][] docTopicWordCounts = new SparseCount[D][K];
        for (int dd = 0; dd < D; dd++) {
            for (int kk = 0; kk < K; kk++) {
                docTopicWordCounts[dd][kk] = new SparseCount();
            }
            for (int nn = 0; nn < words[dd].length; nn++) {
                docTopicWordCounts[dd][z[dd][nn].issueIndex].increment(words[dd][nn]);
            }
        }

        for (int kk = 0; kk < K; kk++) {
            SparseVector[] lexDesginMatrix = new SparseVector[A];
            double[] lexResponses = new double[A];
            for (int aa = 0; aa < A; aa++) {
                lexDesginMatrix[aa] = new SparseVector(V);
            }

            for (int dd = 0; dd < D; dd++) {
                int aa = authors[dd];
                for (int vv : docTopicWordCounts[dd][kk].getIndices()) {
                    int count = docTopicWordCounts[dd][kk].getCount(vv);
                    lexDesginMatrix[aa].change(vv, (double) count / authorTotalWordWeights[aa]);
                    lexicalVals[aa] -= count * tau[kk].get(vv) / authorTotalWordWeights[aa];
                }
            }

            for (int aa = 0; aa < A; aa++) {
                lexResponses[aa] = u[aa] - topicVals[aa] - lexicalVals[aa];
            }

            RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                    lexResponses, tau[kk].dense(), lexDesginMatrix, rho, 0.0, lambda);
            LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
            boolean converged = false;
            try {
                converged = optimizer.optimize();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            double[] topicTau = new double[V];
            for (int vv = 0; vv < V; vv++) {
                topicTau[vv] = optimizable.getParameter(vv);
            }
            tau[kk] = new SparseVector(topicTau);

            // update
            for (int aa = 0; aa < A; aa++) {
                lexicalVals[aa] += lexDesginMatrix[aa].dotProduct(tau[kk]);
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    protected long updateTausOWLQN() {
        if (isReporting) {
            logln("+++ Updating tau using OWL-QN ...");
        }
        long sTime = System.currentTimeMillis();

        SparseCount[][] docTopicWordCounts = new SparseCount[D][K];
        for (int dd = 0; dd < D; dd++) {
            for (int kk = 0; kk < K; kk++) {
                docTopicWordCounts[dd][kk] = new SparseCount();
            }
            for (int nn = 0; nn < words[dd].length; nn++) {
                docTopicWordCounts[dd][z[dd][nn].issueIndex].increment(words[dd][nn]);
            }
        }

        for (int kk = 0; kk < K; kk++) {
            SparseVector[] lexDesginMatrix = new SparseVector[A];
            double[] lexResponses = new double[A];
            for (int aa = 0; aa < A; aa++) {
                lexDesginMatrix[aa] = new SparseVector(V);
            }

            for (int dd = 0; dd < D; dd++) {
                int aa = authors[dd];
                for (int vv : docTopicWordCounts[dd][kk].getIndices()) {
                    int count = docTopicWordCounts[dd][kk].getCount(vv);
                    lexDesginMatrix[aa].change(vv, (double) count / authorTotalWordWeights[aa]);
                    lexicalVals[aa] -= count * tau[kk].get(vv) / authorTotalWordWeights[aa];
                }
            }

            for (int aa = 0; aa < A; aa++) {
                lexResponses[aa] = u[aa] - topicVals[aa] - lexicalVals[aa];
            }

            OWLQNLinearRegression opt = new OWLQNLinearRegression(basename, l1, l2);
            opt.setQuiet(true);
            OWLQNLinearRegression.setVerbose(false);
            double[] topicTau = tau[kk].dense();
            opt.train(lexDesginMatrix, lexResponses, topicTau);
            tau[kk] = new SparseVector(topicTau);

            // update
            for (int aa = 0; aa < A; aa++) {
                lexicalVals[aa] += lexDesginMatrix[aa].dotProduct(tau[kk]);
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    @Override
    protected void updateUs() {
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
            grad -= (u[aa] - topicVals[aa] - lexicalVals[aa]) / rho; // prior
            u[aa] += aRate * grad; // update
        }
    }

    @Override
    public double getLogLikelihood() {
        return 0.0;
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

        // train author scores
        StringBuilder authorStr = new StringBuilder();
        authorStr.append(A).append("\n");
        for (int aa = 0; aa < A; aa++) {
            authorStr.append(u[aa]).append("\n");
        }
        // train bill scores
        StringBuilder billStr = new StringBuilder();
        billStr.append(B).append("\n");
        for (int bb = 0; bb < B; bb++) {
            billStr.append(x[bb]).append("\n");
            billStr.append(y[bb]).append("\n");
        }

        // model string
        StringBuilder modelStr = new StringBuilder();
        for (int vv = 0; vv < V; vv++) {
            modelStr.append(vv).append("\t").append(this.wordWeights[vv]).append("\n");
        }
        for (int kk = 0; kk < K; kk++) {
            modelStr.append(SparseVector.output(tau[kk])).append("\n");
        }

        Stack<Node> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pop();
            modelStr.append(Integer.toString(node.born)).append("\n");
            modelStr.append(Integer.toString(node.issueIndex)).append("\n");
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

    @Override
    public void inputState(String filepath) {
        if (verbose) {
            logln("--- Reading state from " + filepath);
        }
        try {
            inputModel(filepath);
            inputAssignments(filepath);
            inputBillIdealPoints(filepath);
            inputAuthorIdealPoints(filepath);
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

            tau = new SparseVector[K];
            for (int kk = 0; kk < K; kk++) {
                tau[kk] = SparseVector.input(reader.readLine());
            }

            HashMap<String, Node> nodeMap = new HashMap<String, Node>();
            String line;
            while ((line = reader.readLine()) != null) {
                int born = Integer.parseInt(line);
                int issueIdx = Integer.parseInt(reader.readLine());
                String pathStr = reader.readLine();
                double nodeEta = Double.parseDouble(reader.readLine());
                double nodePi = Double.parseDouble(reader.readLine());
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

                Node node = new Node(born, nodeIndex, nodeLevel, topic, parent, issueIdx, nodeEta, false);
                node.pi = nodePi;
                node.theta = theta;
                node.tokenCounts = tokenCounts;
                node.subtreeTokenCounts = subtreeTokenCounts;
                node.changeStatus();
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

            // update path eta
            Queue<Node> queue = new LinkedList<>();
            queue.add(root);
            while (!queue.isEmpty()) {
                Node node = queue.poll();
                for (Node child : node.getChildren()) {
                    queue.add(child);
                }
                node.updatePathEta();
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

            lexicalVals = new double[A];
            topicVals = new double[A];
            for (int dd = 0; dd < D; dd++) {
                int aa = authors[dd];
                for (int nn = 0; nn < words[dd].length; nn++) {
                    topicVals[aa] += z[dd][nn].pathEta / authorTotalWordWeights[aa];
                    lexicalVals[aa] += tau[z[dd][nn].issueIndex].get(words[dd][nn])
                            / authorTotalWordWeights[aa];
                }
            }
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
    @Override
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

            String[] topWords = getTopWords(node.getPhi(), numWords);

            for (int i = 0; i < node.getLevel(); i++) {
                str.append("   ");
            }
            str.append(node.getPathString())
                    .append(" (").append(node.born)
                    .append("; ").append(node.issueIndex)
                    .append("; ").append(node.getContent().getCountSum())
                    .append("; ").append(MiscUtils.formatDouble(node.eta))
                    .append("; ").append(MiscUtils.formatDouble(node.pathEta))
                    .append("; ").append(node.tokenCounts.getCountSum())
                    .append("; ").append(node.subtreeTokenCounts.getCountSum())
                    .append(")");
            if (node.getLevel() == 1) {
                if (labelVocab != null && labelVocab.size() == K) {
                    str.append(" ").append(labelVocab.get(node.getIndex()));
                }
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
            str.append(node.getTopObservations()).append("\n");

            if (node.getLevel() == 1) {
                // ranked lexical items
                ArrayList<RankingItem<Integer>> rankLexs = new ArrayList<>();
                for (int vv : tau[node.issueIndex].getIndices()) {
                    rankLexs.add(new RankingItem<Integer>(vv, tau[node.issueIndex].get(vv)));
                }
                Collections.sort(rankLexs);

                // positive
                for (int i = 0; i < node.getLevel(); i++) {
                    str.append("   ");
                }
                str.append("+++ ");
                for (int jj = 0; jj < Math.min(10, rankLexs.size()); jj++) {
                    RankingItem<Integer> rankLex = rankLexs.get(jj);
                    if (rankLex.getPrimaryValue() > 0) {
                        str.append(wordVocab.get(rankLex.getObject()))
                                .append(" (").append(MiscUtils.formatDouble(rankLex.getPrimaryValue()))
                                .append("), ");
                    }
                }
                str.append("\n");

                // negative
                for (int i = 0; i < node.getLevel(); i++) {
                    str.append("   ");
                }
                str.append("--- ");
                for (int jj = 0; jj < Math.min(10, rankLexs.size()); jj++) {
                    RankingItem<Integer> rankLex = rankLexs.get(rankLexs.size() - 1 - jj);
                    if (rankLex.getPrimaryValue() < 0) {
                        str.append(wordVocab.get(rankLex.getObject()))
                                .append(" (").append(MiscUtils.formatDouble(rankLex.getPrimaryValue()))
                                .append("), ");
                    }
                }
                str.append("\n");
            }
            str.append("\n");
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

    public void analyzeAuthors(File authorAnalysisFile,
            HashMap<String, Author> authorTable) {
        if (verbose) {
            logln("Analyzing authors to " + authorAnalysisFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(authorAnalysisFile);
            writer.write("Legislator\tFWID\tIdealPoint\tTopicIdealPoint\tLexicalIdealPoint\n");
            for (int aa = 0; aa < A; aa++) {
                String authorId = authorVocab.get(authorIndices.get(aa));
                String authorName = authorTable.get(authorId).getProperty(GTLegislator.NAME);
                String authorFWID = authorTable.get(authorId).getProperty(GTLegislator.FW_ID);
                writer.write(authorName + "\t" + authorFWID + "\t" + (u == null ? "NA" : u[aa])
                        + "\t" + topicVals[aa] + "\t" + lexicalVals[aa]
                        + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + authorAnalysisFile);
        }
    }

    class Node extends TreeNode<Node, DirMult> {

        protected final int born;
        protected final boolean extensible;
        protected final int issueIndex;
        protected SparseCount subtreeTokenCounts;
        protected SparseCount tokenCounts;
        protected double eta; // regression parameter
        protected double pi;
        private HashMap<Integer, Double> theta;
        protected double pathEta;
        protected boolean isNew;

        // estimated topics after training, which is used for test
        protected double[] phihat;

        public Node(int iter, int index, int level, DirMult content, Node parent,
                int issueIndex, double eta, boolean extensible) {
            super(index, level, content, parent);
            this.born = iter;
            this.issueIndex = issueIndex;
            this.eta = eta;
            this.subtreeTokenCounts = new SparseCount();
            this.tokenCounts = new SparseCount();
            this.extensible = extensible;
            this.isNew = true;
            this.pi = getPi();
            this.theta = new HashMap<>();
            if (this.extensible) {
                this.theta.put(NEW_CHILD_INDEX, 1.0);
            }
        }

        void changeStatus() {
            this.isNew = false;
        }

        void updatePathEta() {
            this.pathEta = 0.0;
            Node tempNode = this;
            while (tempNode != null) {
                this.pathEta += tempNode.eta;
                tempNode = tempNode.parent;
            }
        }

        /**
         * Update global pi.
         */
        void updatePi() {
            if (this.level == L - 1) {
                return;
            }
            int totalStay = this.tokenCounts.getCountSum();
            int totalStayAndPass = this.subtreeTokenCounts.getCountSum();
            double gamma = getGamma(level);
            this.pi = (totalStay + gamma * getPi()) / (totalStayAndPass + gamma);
        }

        /**
         * Update theta (i.e., distribution over children) of this node with new
         * counts.
         */
        void updateTheta() {
            if (this.isLeaf() && !this.extensible) {
                if (!this.theta.isEmpty()) {
                    throw new RuntimeException("Non-empty theta at non-extensible leaf node");
                }
                return;
            }
            double gAlpha = getGlobalAlpha(level);

            // update counts
            SparseCount approxThetaCounts = new SparseCount();
            for (Node child : getChildren()) {
                int childIdx = child.getIndex();
                for (int dd : child.subtreeTokenCounts.getIndices()) {
                    int rawCount = child.subtreeTokenCounts.getCount(dd);
                    Double curTheta = this.theta.get(childIdx);
                    if (curTheta == null) {
                        curTheta = 1.0 / this.getNumChildren();
                    }
                    int approxCount = getApproxCount(rawCount, curTheta);
                    approxThetaCounts.changeCount(childIdx, approxCount);
                }
            }

            if (approxThetaCounts.size() != this.getNumChildren()) {
                throw new MismatchRuntimeException(approxThetaCounts.size(), this.getNumChildren());
            }

            this.theta = new HashMap<>();
            double norm = approxThetaCounts.getCountSum() + gAlpha;
            for (int childIdx : approxThetaCounts.getIndices()) {
                this.theta.put(childIdx, (double) approxThetaCounts.getCount(childIdx) / norm);
            }
            if (this.extensible) {
                this.theta.put(NEW_CHILD_INDEX, gAlpha / norm);
            }
        }

        void setTheta(HashMap<Integer, Double> theta) {
            this.theta = theta;
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

        /**
         * Return the number of tokens of a given document which are assigned to
         * any nodes below this node.
         *
         * @param dd Document index
         */
        int getPassingCount(int dd) {
            return subtreeTokenCounts.getCount(dd) - tokenCounts.getCount(dd);
        }

        double[] getPhi() {
            if (this.phihat == null) {
                double[] phi = new double[V];
                for (int vv = 0; vv < V; vv++) {
                    phi[vv] = getContent().getProbability(vv);
                }
                return phi;
            }
            return phihat;
        }

        boolean isEmpty() {
            return this.getContent().isEmpty();
        }

        String[] getTopWords(int numTopWords) {
            ArrayList<RankingItem<String>> topicSortedVocab
                    = IOUtils.getSortedVocab(getPhi(), wordVocab);
            String[] topWords = new String[numTopWords];
            for (int i = 0; i < numTopWords; i++) {
                topWords[i] = topicSortedVocab.get(i).getObject();
            }
            return topWords;
        }

        String getTopObservations() {
            return MiscUtils.getTopObservations(wordVocab, getContent().getSparseCounts(), 10);
        }

        public void validate(String msg) {
            this.content.validate(msg);
            this.tokenCounts.validate(msg);
            this.subtreeTokenCounts.validate(msg);
            if (!this.isRoot()) {
                if (this.pathEta != this.parent.pathEta + this.eta) {
                    throw new RuntimeException(msg
                            + ". this.pathEta = " + this.pathEta
                            + ". sum = " + (this.parent.pathEta + this.eta));
                }
            }

            if (extensible) {
                if (this.theta.size() != this.getNumChildren() + 1) {
                    throw new MismatchRuntimeException(theta.size(), (getNumChildren() + 1));
                }
            } else {
                if (this.theta.size() != this.getNumChildren()) {
                    throw new MismatchRuntimeException(theta.size(), getNumChildren());
                }
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder();
            str.append("[").append(getPathString());
            str.append(", ").append(born);
            str.append(", ").append(issueIndex);
            str.append(", c (").append(getChildren().size()).append(")");
            // word types
            str.append(", (").append(getContent().getCountSum()).append(")");
            str.append(", ").append(MiscUtils.formatDouble(eta));
            str.append(", ").append(MiscUtils.formatDouble(pathEta));
            str.append(", ").append(MiscUtils.formatDouble(pi));
            str.append(", ").append(tokenCounts.getCountSum());
            str.append(", ").append(subtreeTokenCounts.getCountSum());
            str.append(", ").append(extensible);
            str.append(", t: ").append(theta.size());
            str.append("]");
            if (this.level == 1 && labelVocab != null) {
                str.append(" ").append(labelVocab.get(index));
            }
            return str.toString();
        }
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

        testBurnIn = 250;
        testMaxIter = 500;
        testSampleLag = 5;
        testRepInterval = 5;
        if (verbose) {
            logln("Setting up test ...");
            logln("--- test burn-in: " + testBurnIn);
            logln("--- test maximum number of iterations: " + testMaxIter);
            logln("--- test sample lag: " + testSampleLag);
            logln("--- test report interval: " + testRepInterval);
        }

        // sample on test data
        ArrayList<SparseVector[]> predictionList = sampleNewDocuments(stateFile, assignmentFile);

        // make prediction on votes of unknown voters
        SparseVector[] predictions = averagePredictions(predictionList);

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
    private ArrayList<SparseVector[]> sampleNewDocuments(
            File stateFile,
            File assignmentFile) {
        if (verbose) {
            logln("--- Sampling on test data using " + stateFile);
        }

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
        inputBillIdealPoints(stateFile.getAbsolutePath());
        initializeDataStructure();

        // debug
        System.out.println(printGlobalTree());

        // sample
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (iter = 0; iter < testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter
                        + "\t @ " + Thread.currentThread().getId();
                if (iter < testBurnIn) {
                    logln("--- Burning in. " + str);
                } else {
                    logln("--- Sampling. " + str);
                }
            }

            // sample topic assignments
            if (iter == 0) {
                sampleZs(!REMOVE, !ADD, !REMOVE, ADD, !OBSERVED, !EXTEND);
            } else {
                sampleZs(!REMOVE, !ADD, REMOVE, ADD, !OBSERVED, !EXTEND);
            }

            if (iter >= testBurnIn && iter % testSampleLag == 0) {
                predictionList.add(predictOutMatrix());
            }
        }

        if (assignmentFile != null) { // output assignments of test data
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
                throw new RuntimeException("Exception while outputing");
            }
        }
        return predictionList;
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
            HybridSNHDPIdealPoint sampler) {
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
                HybridSNHDPTestRunner runner = new HybridSNHDPTestRunner(
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

class HybridSNHDPTestRunner implements Runnable {

    HybridSNHDPIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public HybridSNHDPTestRunner(HybridSNHDPIdealPoint sampler,
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
        HybridSNHDPIdealPoint testSampler = new HybridSNHDPIdealPoint();
        testSampler.setVerbose(true);
        testSampler.setDebug(false);
        testSampler.setLog(false);
        testSampler.setReport(false);
        testSampler.configure(sampler);
        testSampler.setWordVocab(sampler.getWordVocab());
        testSampler.setTestConfigurations(sampler.getBurnIn(),
                sampler.getMaxIters(), sampler.getSampleLag());
        testSampler.setupData(testDocIndices, testWords, testAuthors, null,
                testAuthorIndices, null, testVotes);
        try {
            testSampler.test(stateFile, predictionFile, null);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
