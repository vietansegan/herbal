package votepredictor.textidealpoint;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Author;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;
import optimization.RidgeLinearRegressionOptimizable;
import sampler.unsupervised.RecursiveLDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
import sampling.util.TreeNode;
import util.IOUtils;
import util.MiscUtils;
import util.MismatchRuntimeException;
import util.RankingItem;
import util.SamplerUtils;
import util.SparseVector;
import util.StatUtils;
import util.evaluation.Measurement;
import util.govtrack.GTLegislator;
import votepredictor.AbstractVotePredictor;

/**
 *
 * @author vietan
 */
public class HybridSNLDAIdealPoint extends AbstractTextSingleIdealPoint {

    public static final int INVALID_ISSUE_INDEX = -1;
    protected double[] alphas;          // [L-1]
    protected double[] betas;           // [L]
    protected double[] sigmas;          // [L-1] eta variance (excluding root)
    protected double[] pis;             // [L-1] mean of bias coins
    protected double[] gammas;          // [L-1] scale of bias coins
    protected double lambda;
    protected double l1;
    protected double l2;
    // input
    protected int K; // number of issues
    protected int J; // number of frames per issue
    protected int L; // number of levels
    protected boolean hasRootTopic;
    // latent
    Node root;
    Node[][] z;

    protected SparseVector[] tau; // each topic has its own lexical regression
    protected double[] topicVals;
    protected double[] lexicalVals;

    // internal
    protected ArrayList<String> labelVocab;
    protected int numTokensAccepted;

    public HybridSNLDAIdealPoint() {
        this.basename = "Hybrid-SNLDA-ideal-point";
    }

    public HybridSNLDAIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setLabelVocab(ArrayList<String> labVoc) {
        this.labelVocab = labVoc;
    }

    public void configure(HybridSNLDAIdealPoint sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.J,
                sampler.alphas,
                sampler.betas,
                sampler.pis,
                sampler.gammas,
                sampler.rho,
                sampler.sigma,
                sampler.sigmas,
                sampler.lambda,
                sampler.hasRootTopic,
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(String folder,
            int V, int K, int J,
            double[] alphas,
            double[] betas,
            double[] pis,
            double[] gammas,
            double rho,
            double sigma, // stadard deviation of Gaussian for regression parameters
            double[] sigmas,
            double lambda,
            boolean hasRootTopic,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }
        this.folder = folder;
        this.V = V;
        this.K = K;
        this.J = J;
        this.L = 3;

        this.alphas = alphas;
        this.betas = betas;
        this.pis = pis;
        this.gammas = gammas;
        this.rho = rho;
        this.sigma = sigma;
        this.sigmas = sigmas;
        this.lambda = lambda;
        this.hasRootTopic = hasRootTopic;
        this.wordWeightType = WordWeightType.NONE;

        this.hyperparams = new ArrayList<Double>();
        for (double alpha : alphas) {
            this.hyperparams.add(alpha);
        }
        for (double beta : betas) {
            this.hyperparams.add(beta);
        }
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

        if (verbose) {
            logln("--- V = " + this.V);
            logln("--- K = " + this.K);
            logln("--- J = " + this.J);
            logln("--- folder\t" + folder);
            logln("--- alphas:\t" + MiscUtils.arrayToString(alphas));
            logln("--- betas:\t" + MiscUtils.arrayToString(betas));
            logln("--- pis:\t" + MiscUtils.arrayToString(pis));
            logln("--- gammas:\t" + MiscUtils.arrayToString(gammas));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- sigmas:\t" + MiscUtils.arrayToString(sigmas));
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- lambda:\t" + MiscUtils.formatDouble(lambda));
            logln("--- has root topic:\t" + hasRootTopic);
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- report interval:\t" + REP_INTERVAL);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + this.initState);
            logln("--- word weight type:\t" + this.wordWeightType);
        }

        if (alphas.length != L - 1) {
            throw new MismatchRuntimeException(this.alphas.length, L - 1);
        }

        if (this.betas.length != L) {
            throw new MismatchRuntimeException(this.betas.length, L);
        }

        if (this.sigmas.length != L - 1) {
            throw new MismatchRuntimeException(this.sigmas.length, L - 1);
        }

        if (this.pis.length != L - 1) {
            throw new MismatchRuntimeException(this.pis.length, L - 1);
        }

        if (this.gammas.length != L - 1) {
            throw new MismatchRuntimeException(this.gammas.length, L - 1);
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
        str.append("_p");
        for (double p : pis) {
            str.append("-").append(MiscUtils.formatDouble(p));
        }
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
    }

    protected double getAlpha(int l) {
        return this.hyperparams.get(l);
    }

    protected double getBeta(int l) {
        return this.hyperparams.get(L - 1 + l);
    }

    protected double getAlpha(ArrayList<Double> params, int l) {
        return params.get(l);
    }

    protected double getBeta(ArrayList<Double> params, int l) {
        return params.get(L - 1 + l);
    }

    protected double getPi(int l) {
        return this.pis[l];
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
        double[] rAlpha = {0.25, 0.1};
        double[] rBetas = {1.0, 0.1};
        int[] Ks = new int[]{K, J};
        RecursiveLDA rlda = runRecursiveLDA(words, Ks, rAlpha, rBetas, V, issuePhis);

        DirMult rootTopic = new DirMult(V, getBeta(0) * V, 1.0 / V);
        this.root = new Node(iter, 0, 0, rootTopic, null, INVALID_ISSUE_INDEX, 0.0);
        for (int kk = 0; kk < K; kk++) {
            DirMult issueTopic = new DirMult(V, getBeta(1) * V,
                    rlda.getTopicWord(new int[]{kk}).getDistribution());
            Node issueNode = new Node(iter, kk, 1, issueTopic, root, kk, 0.0);
            root.addChild(kk, issueNode);
            for (int jj = 0; jj < J; jj++) {
                DirMult frameTopic = new DirMult(V, getBeta(2) * V,
                        rlda.getTopicWord(new int[]{kk, jj}).getDistribution());
                Node frameNode = new Node(iter, jj, 2, frameTopic, issueNode, kk, 0.0);
                issueNode.addChild(jj, frameNode);
            }
        }
        this.root.initializeGlobalTheta();
        this.root.initializeGlobalPi();
        for (Node issueNode : this.root.getChildren()) {
            issueNode.initializeGlobalTheta();
            issueNode.initializeGlobalPi();
        }

        tau = new SparseVector[K];
        for (int kk = 0; kk < K; kk++) {
            tau[kk] = new SparseVector(V);
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
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
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
        if (this.lambda > 0) {
            updateTausLBFGS();
        } else {
            updateTausOWLQN();
        }
        updateEtas();
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

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

//        boolean condition = dd == 1;
        int level = curNode.getLevel();
        double lAlpha = getAlpha(level);
        double gamma = getGamma(level);

        double stayprob = 0.0;
        if (hasRootTopic || (!hasRootTopic && !curNode.isRoot())) {
            stayprob = (curNode.tokenCounts.getCount(dd) + gamma * curNode.pi)
                    / (curNode.subtreeTokenCounts.getCount(dd) + gamma);
        }
        double passprob = 1.0 - stayprob;

        // debug
//        if (condition) {
//            System.out.println("nn = " + nn + ". " + curNode.toString());
//        }
        int KK = curNode.getNumChildren();
        double[] probs = new double[KK + 1];
        double norm = curNode.getPassingCount(dd) + lAlpha * KK;
        for (Node child : curNode.getChildren()) {
            int kk = child.getIndex();
            double pathprob = (child.subtreeTokenCounts.getCount(dd)
                    + lAlpha * KK * curNode.theta[kk]) / norm;
            double wordprob = child.getPhi(words[dd][nn]);
            probs[kk] = passprob * pathprob * wordprob;

            // debug
//            if (condition) {
//                System.out.println("    kk = " + kk
//                        + ". theta: " + curNode.theta[kk]
//                        + ". count: " + child.subtreeTokenCounts.getCount(dd)
//                        + ". " + passprob
//                        + ". " + pathprob
//                        + ". " + wordprob
//                        + ". " + probs[kk]);
//            }
        }
        double wordprob = curNode.getPhi(words[dd][nn]);
        probs[KK] = stayprob * wordprob;

        // debug
//        if (condition) {
//            System.out.println("    KK = " + KK
//                    + ". " + stayprob
//                    + ". " + wordprob
//                    + ". " + probs[KK]);
//        }
        int sampledIdx = SamplerUtils.scaleSample(probs);

        // debug
//        if (condition) {
//            System.out.println("--- >>> " + sampledIdx + "\n");
//        }
//        if (dd == 10) {
//            System.exit(1);
//        }
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
        double gamma = getGamma(level);
        double stayprob = (source.tokenCounts.getCount(dd) + gamma * source.pi)
                / (source.subtreeTokenCounts.getCount(dd) + gamma);
        double passprob = 1.0 - stayprob;

        double pNum = 0.0;
        double pDen = 0.0;
        double aNum = 0.0;
        double aDen = 0.0;
        double norm = source.getPassingCount(dd) + lAlpha * KK;
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
        logln(msg + ". Validation not implemented!");
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
//            if (node.isEmpty()) { // skip empty nodes
//                continue;
//            }

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
                    .append(" [").append(nodeCount).append("]")
                    .append(" [").append(obsCount)
                    .append(", ").append(MiscUtils.formatDouble((double) obsCount / nodeCount)).append("]")
                    .append(" [").append(subtreeObsCount)
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

                Node node = new Node(born, nodeIndex, nodeLevel, topic, parent, issueIdx, eta);
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
        protected final int issueIndex;
        protected SparseCount subtreeTokenCounts;
        protected SparseCount tokenCounts;
        protected double eta; // regression parameter
        protected double pi;
        protected double[] theta;
        protected double pathEta;

        // estimated topics after training, which is used for test
        protected double[] phihat;

        public Node(int iter, int index, int level, DirMult content, Node parent,
                int issueIndex, double eta) {
            super(index, level, content, parent);
            this.born = iter;
            this.issueIndex = issueIndex;
            this.eta = eta;
            this.subtreeTokenCounts = new SparseCount();
            this.tokenCounts = new SparseCount();
        }

        void updatePathEta() {
            this.pathEta = 0.0;
            Node tempNode = this;
            while (tempNode != null) {
                this.pathEta += tempNode.eta;
                tempNode = tempNode.parent;
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

        int getPassingCount(int dd) {
            return subtreeTokenCounts.getCount(dd) - tokenCounts.getCount(dd);
        }

        boolean isEmpty() {
            return this.getContent().isEmpty();
        }

        void initializeGlobalPi() {
            this.pi = getPi(level);
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
            str.append(", ").append(MiscUtils.formatDouble(pi));
            str.append(", ").append(tokenCounts.getCountSum());
            str.append(", ").append(subtreeTokenCounts.getCountSum());
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

        // sample
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (iter = 0; iter < testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                String str = "Iter " + iter + "/" + testMaxIter;
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
            HybridSNLDAIdealPoint sampler) {
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
                HybridSNLDATestRunner runner = new HybridSNLDATestRunner(
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

class HybridSNLDATestRunner implements Runnable {

    HybridSNLDAIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public HybridSNLDATestRunner(HybridSNLDAIdealPoint sampler,
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
        HybridSNLDAIdealPoint testSampler = new HybridSNLDAIdealPoint();
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
