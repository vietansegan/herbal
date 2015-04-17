package votepredictor.textidealpoint.hierarchy;

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
import optimization.RidgeLinearRegressionOptimizable;
import sampler.unsupervised.LDA;
import sampling.likelihood.DirMult;
import sampling.util.SparseCount;
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
import votepredictor.textidealpoint.AbstractTextSingleIdealPoint;

/**
 *
 * @author vietan
 */
public class HierMultiTIPM extends AbstractTextSingleIdealPoint {

    protected double topicAlpha;
    protected double frameAlpha;
    protected double topicBeta;
    protected double frameBeta;

    protected double gamma; // eta variance
    protected double lambda; // trade-off between single ideal point and text-based
    protected double epsilon; // initial gradient descent rate

    // input
    protected int K;
    protected int J; // number of frames
    protected double[][] priorTopics;
    protected int[][] billWords;

    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[][] frameWords;
    protected DirMult[] docTopics;
    protected DirMult[][] docFrames;

    // debates
    protected int[][] topicZs;
    protected int[][] frameZs;
    protected double[][] etas; // regression parameters for topics
    protected SparseCount[][] frameCounts;  // per author
    protected SparseCount[] topicCounts;    // per author

    // bills
    protected DirMult[] billTopics;
    protected int[][] billZs;
    protected SparseVector[] billThetas;
    protected double[][] billTopicPriors;

    protected double[][] us;

    protected ArrayList<double[][]> etaList;
    private int numFrameAssignmentChange;
    private ArrayList<String> topicVocab;
    private double sqrtRho;

    private ArrayList<Integer> authorList;
    private ArrayList<Integer> billList;

    public HierMultiTIPM() {
        this.basename = "Hier-Mult-TIPM";
    }

    public HierMultiTIPM(String bname) {
        this.basename = bname;
    }

    public void setTopicVocab(ArrayList<String> topicVoc) {
        this.topicVocab = topicVoc;
    }

    public void setBillWords(int[][] billWords) {
        this.billWords = billWords;

        if (this.billWords.length != B) {
            throw new MismatchRuntimeException(this.billWords.length, B);
        }
    }

    public void setBillTopicPriors(double[][] billTopicPriors) {
        this.billTopicPriors = billTopicPriors;
    }

    public double[][] getBillThetas() {
        double[][] thetas = new double[B][];
        for (int bb = 0; bb < B; bb++) {
            thetas[bb] = this.billThetas[bb].dense();
        }
        return thetas;
    }

    public void configure(HierMultiTIPM sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.J,
                sampler.topicAlpha,
                sampler.frameAlpha,
                sampler.topicBeta,
                sampler.frameBeta,
                sampler.rho,
                sampler.sigma,
                sampler.gamma,
                sampler.lambda,
                sampler.epsilon,
                sampler.priorTopics,
                sampler.initState,
                sampler.paramOptimized,
                sampler.BURN_IN,
                sampler.MAX_ITER,
                sampler.LAG,
                sampler.REP_INTERVAL);
    }

    public void configure(
            String folder,
            int V, int K, int J,
            double topicAlpha, double frameAlpha,
            double topicBeta, double frameBeta,
            double rho,
            double sigma,
            double gamma,
            double lambda,
            double epsilon,
            double[][] priorTopics,
            InitialState initState,
            boolean paramOpt,
            int burnin, int maxiter, int samplelag, int repInt) {
        if (verbose) {
            logln("Configuring ...");
        }

        this.folder = folder;
        this.K = K;
        this.J = J;
        this.V = V;

        this.hyperparams = new ArrayList<Double>();
        this.topicAlpha = topicAlpha;
        this.frameAlpha = frameAlpha;
        this.topicBeta = topicBeta;
        this.frameBeta = frameBeta;

        this.rho = rho;
        this.sqrtRho = Math.sqrt(rho);
        this.sigma = sigma;
        this.gamma = gamma;
        this.lambda = lambda;
        this.epsilon = epsilon;
        this.priorTopics = priorTopics;
        this.wordWeightType = WordWeightType.NONE;

        this.sampledParams = new ArrayList<ArrayList<Double>>();
        this.sampledParams.add(cloneHyperparameters());

        this.BURN_IN = burnin;
        this.MAX_ITER = maxiter;
        this.LAG = samplelag;
        this.REP_INTERVAL = repInt;

        this.initState = initState;
        this.paramOptimized = paramOpt;
        this.prefix += initState.toString();
        this.report = true;

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(this.K)
                .append("_J-").append(this.J)
                .append("_ta-").append(formatter.format(this.topicAlpha))
                .append("_fa-").append(formatter.format(this.frameAlpha))
                .append("_tb-").append(formatter.format(this.topicBeta))
                .append("_fb-").append(formatter.format(this.frameBeta))
                .append("_r-").append(formatter.format(this.rho))
                .append("_s-").append(formatter.format(this.sigma))
                .append("_g-").append(formatter.format(this.gamma))
                .append("_l-").append(formatter.format(this.lambda))
                .append("_e-").append(MiscUtils.formatDouble(this.epsilon, 10));
        str.append("_opt-").append(this.paramOptimized);
        str.append("_prior-").append(this.priorTopics != null);
        this.name = str.toString();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + this.K);
            logln("--- num frames per topic:\t" + this.J);
            logln("--- num word types:\t" + this.V);
            logln("--- topic alpha:\t" + MiscUtils.formatDouble(this.topicAlpha));
            logln("--- frame alpha:\t" + MiscUtils.formatDouble(this.frameAlpha));
            logln("--- topic beta:\t" + MiscUtils.formatDouble(this.topicBeta));
            logln("--- frame beta:\t" + MiscUtils.formatDouble(this.frameBeta));
            logln("--- rho:\t" + MiscUtils.formatDouble(this.rho));
            logln("--- sqrt rho:\t" + MiscUtils.formatDouble(this.sqrtRho));
            logln("--- sigma:\t" + MiscUtils.formatDouble(this.sigma));
            logln("--- gamma:\t" + MiscUtils.formatDouble(this.gamma));
            logln("--- lambda:\t" + MiscUtils.formatDouble(this.lambda));
            logln("--- epsilon:\t" + MiscUtils.formatDouble(this.epsilon));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- has prior? " + (priorTopics != null));
        }
    }

    public double[][] getMultiUs() {
        return this.us;
    }

    public double[][] getEstimatedUs() {
        double[][] estUs = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                estUs[aa][kk] = getEstimatedU(aa, kk);
            }
        }
        return estUs;
    }

    @Override
    public void initialize() {
        initialize(priorTopics);
    }

    public void initialize(double[][] seededTopics) {
        if (verbose) {
            logln("Initializing ...");
        }
        iter = INIT;
        isReporting = true;

        initializeIdealPoint();
        initializeBillThetas();
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

    protected void initializeBillThetas() {
        LDA lda = runLDA(billWords, K, V, billTopicPriors, priorTopics,
                0.5, 1000, 250, 500, 50);
        this.billTopics = lda.getDocTopics();
        this.billZs = lda.getZs();

        // empirical distributions over topics
        this.billThetas = new SparseVector[B];
        for (int bb = 0; bb < B; bb++) {
            this.billThetas[bb] = new SparseVector(K);
            for (int kk : this.billTopics[bb].getSparseCounts().getIndices()) {
                int count = this.billTopics[bb].getCount(kk);
                this.billThetas[bb].set(kk, (double) count / billWords[bb].length);
            }
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
                topicWords[k] = new DirMult(V, topicBeta * V, seededTopics[k]);
            } else {
                topicWords[k] = new DirMult(V, topicBeta * V, 1.0 / V);
            }
        }

        frameWords = new DirMult[K][J];
        for (int kk = 0; kk < K; kk++) {
            for (int jj = 0; jj < J; jj++) {
                frameWords[kk][jj] = new DirMult(V, frameBeta * V, 1.0 / V);
            }
        }

        etas = new double[K][J];
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++) {
                etas[k][j] = SamplerUtils.getGaussian(0.0, gamma);
            }
        }
    }

    protected void initializeDataStructure() {
        topicZs = new int[D][];
        frameZs = new int[D][];
        for (int dd = 0; dd < D; dd++) {
            topicZs[dd] = new int[words[dd].length];
            frameZs[dd] = new int[words[dd].length];
        }

        docTopics = new DirMult[D];
        docFrames = new DirMult[D][K];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, topicAlpha * K, 1.0 / K);
            for (int k = 0; k < K; k++) {
                docFrames[d][k] = new DirMult(J, frameAlpha * J, 1.0 / J);
            }
        }

        if (u != null) {
            us = new double[A][K];
            for (int aa = 0; aa < A; aa++) {
                Arrays.fill(us[aa], u[aa]);
            }
        }

        topicCounts = new SparseCount[A];
        frameCounts = new SparseCount[A][K];
        for (int aa = 0; aa < A; aa++) {
            topicCounts[aa] = new SparseCount();
            for (int kk = 0; kk < K; kk++) {
                frameCounts[aa][kk] = new SparseCount();
            }
        }

        authorList = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            authorList.add(aa);
        }

        billList = new ArrayList<>();
        for (int bb = 0; bb < B; bb++) {
            billList.add(bb);
        }
    }

    protected void initializeAssignments() {
        switch (initState) {
            case RANDOM:
                initializeRandomAssignments();
                break;
            default:
                throw new RuntimeException("Initialization " + initState + " not supported");
        }
    }

    protected void initializeRandomAssignments() {
        if (verbose) {
            logln("--- Initializing random assignments ...");
        }
        sampleZs(!REMOVE, ADD, !REMOVE, ADD, !OBSERVED);
    }

    /**
     * Make prediction on held-out votes of known legislators and known votes.
     *
     * @return Predicted probabilities
     */
    @Override
    public SparseVector[] predictInMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    @Override
    public SparseVector[] predictOutMatrix() {
        SparseVector[] predictions = new SparseVector[validVotes.length];
        for (int aa = 0; aa < A; aa++) {
            int author = authorIndices.get(aa);
            predictions[author] = new SparseVector(validVotes[author].length);
            for (int bb = 0; bb < B; bb++) {
                int bill = billIndices.get(bb);
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb];
                    for (int kk : billThetas[bb].getIndices()) {
                        dotprod += x[bb] * billThetas[bb].get(kk) * getEstimatedU(aa, kk);
                    }
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    protected double getEstimatedU(int aa, int kk) {
        double estUak = u[aa]; // backoff to single ideal point
        if (!frameCounts[aa][kk].isEmpty()) {
            estUak = frameCounts[aa][kk].dotprod(etas[kk]) / frameCounts[aa][kk].getCountSum();
        }
        return estUak;
    }

    @Override
    public String getCurrentState() {
        String str = "\n\nIter " + iter + "/" + MAX_ITER
                + "\t @ " + Thread.currentThread().getId();
        if (iter < BURN_IN) {
            logln("--- Burning in. " + str);
        } else {
            logln("--- Sampling. " + str);
        }
        logln("--- Evaluating ...");
        SparseVector[] predictions = predictInMatrix();
        ArrayList<Measurement> measurements = AbstractVotePredictor.evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> i >>> " + m.getName() + ": " + m.getValue());
        }

        predictions = predictOutMatrix();
        measurements = AbstractVotePredictor.evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> o >>> " + m.getName() + ": " + m.getValue());
        }

        // author mse
        double mse = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                double diff = us[aa][kk] - getEstimatedU(aa, kk);
                mse += diff * diff;
            }
        }
        logln("--- Regression MSE: " + mse / (A * K));

        return str;
    }

    @Override
    public void iterate() {
        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

        updateEtas();

        if (iter > BURN_IN && iter % LAG == 0) {
            updateUXY();
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
        numFrameAssignmentChange = 0;
        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    topicWords[topicZs[d][n]].decrement(words[d][n]);
                    frameWords[topicZs[d][n]][frameZs[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(topicZs[d][n]);
                    docFrames[d][topicZs[d][n]].decrement(frameZs[d][n]);
                    topicCounts[aa].decrement(topicZs[d][n]);
                    frameCounts[aa][topicZs[d][n]].decrement(frameZs[d][n]);
                }

                // sample topic
                double[] probs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    probs[kk] = (docTopics[d].getCount(kk) + topicAlpha * K * docTopics[d].getCenterElement(kk))
                            * topicWords[kk].getProbability(words[d][n]);
                }
                int st = SamplerUtils.scaleSample(probs); // samples topic

                // sample frame
                double[] logprobs = new double[J];
                for (int jj = 0; jj < J; jj++) {
                    logprobs[jj]
                            = Math.log(docFrames[d][st].getCount(jj) + frameAlpha)
                            + Math.log(frameWords[st][jj].getProbability(words[d][n]));
                    if (observe) {
                        double lexMean = (frameCounts[aa][st].dotprod(etas[st]) + etas[st][jj])
                                / (frameCounts[aa][st].getCountSum() + 1);
                        double mean = lambda * u[aa] + (1 - lambda) * lexMean;
                        double resLlh = StatUtils.logNormalProbability(us[aa][st], mean, sqrtRho);
                        logprobs[jj] += resLlh;
                    }
                }
                int sf = SamplerUtils.logMaxRescaleSample(logprobs); // sampled frame

                if (topicZs[d][n] != st) {
                    numTokensChanged++;
                }
                if (topicZs[d][n] != st || frameZs[d][n] != sf) {
                    numFrameAssignmentChange++;
                }
                topicZs[d][n] = st;
                frameZs[d][n] = sf;

                if (addToModel) {
                    topicWords[topicZs[d][n]].increment(words[d][n]);
                    frameWords[topicZs[d][n]][frameZs[d][n]].increment(words[d][n]);
                }
                if (addToData) {
                    docTopics[d].increment(topicZs[d][n]);
                    docFrames[d][topicZs[d][n]].increment(frameZs[d][n]);
                    topicCounts[aa].increment(topicZs[d][n]);
                    frameCounts[aa][topicZs[d][n]].increment(frameZs[d][n]);
                }
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
            logln("--- --- # tokens: " + numTokens
                    + ". # topic asgns: " + numTokensChanged
                    + " (" + MiscUtils.formatDouble((double) numTokensChanged / numTokens) + ")"
                    + ". # frame asgns: " + numFrameAssignmentChange
                    + " (" + MiscUtils.formatDouble((double) numFrameAssignmentChange / numTokens) + ")");
        }
        return eTime;
    }

    /**
     * Update all eta vectors.
     */
    protected long updateEtas() {
        if (isReporting) {
            logln("+++ Updating etas ...");
        }
        long sTime = System.currentTimeMillis();

        int numConverged = 0;
        for (int kk = 0; kk < K; kk++) {
            boolean converged = updateEta(kk);
            if (converged) {
                numConverged++;
            }
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- # converged: " + numConverged + " / " + K);
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    /**
     * Update eta in a subtree.
     *
     * @param kk
     */
    private boolean updateEta(int kk) {
        ArrayList<SparseVector> designMatrixList = new ArrayList<>();
        ArrayList<Double> responseList = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            if (frameCounts[aa][kk].isEmpty()) {
                continue;
            }

            
            // TODO: update this
//            responseList.add(us[aa][kk] - lambda * u[aa]);
            
            responseList.add(us[aa][kk]);
            SparseVector vec = new SparseVector(J);
            for (int jj : frameCounts[aa][kk].getIndices()) {
                double val = (double) frameCounts[aa][kk].getCount(jj)
                        / frameCounts[aa][kk].getCountSum();
                vec.set(jj, val);
            }
            designMatrixList.add(vec);
        }

        int numA = responseList.size();
        SparseVector[] designMatrix = new SparseVector[numA];
        double[] responses = new double[numA];
        for (int aa = 0; aa < numA; aa++) {
            designMatrix[aa] = designMatrixList.get(aa);
            responses[aa] = responseList.get(aa);
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, etas[kk], designMatrix, rho, 0.0, gamma);
        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update regression parameters
        for (int jj = 0; jj < J; jj++) {
            etas[kk][jj] = optimizable.getParameter(jj);
        }

        return converged;
    }

    @Override
    protected long updateUXY() {
        if (isReporting) {
            logln("+++ Updating UXY ...");
        }
        long sTime = System.currentTimeMillis();
        for (int step = 0; step < numSteps; step++) {
            updateUs();
            updateXYs();
        }
        long eTime = System.currentTimeMillis() - sTime;

        if (isReporting) {
            double aMSE = 0.0;
            for (int aa = 0; aa < A; aa++) {
                for (int kk = 0; kk < K; kk++) {
                    double diff = us[aa][kk] - u[aa];
                    aMSE += diff * diff;
                }
            }
            logln("--- --- Author MSE: " + (aMSE / A));
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    @Override
    protected double getLearningRate() {
        return this.epsilon;
    }

    @Override
    protected void updateUs() {
        double aRate = getLearningRate();
        Collections.shuffle(authorList);
        for (int aa : authorList) {
            if (!validAs[aa] || authorTotalWordWeights[aa] == 0) {
                continue;
            }
            double[] grads = new double[K];
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    for (int kk : billThetas[bb].getIndices()) {
                        grads[kk] += x[bb] * billThetas[bb].get(kk) * (getVote(aa, bb) - prob);
                    }
                }
            }
            for (int kk = 0; kk < K; kk++) {
                double mean = lambda * u[aa] + (1 - lambda) * getEstimatedU(aa, kk);
                grads[kk] -= (us[aa][kk] - mean) / rho;
                us[aa][kk] += aRate * grads[kk];
            }
        }
    }

    @Override
    public void updateXYs() {
        double bRate = getLearningRate();
        Collections.shuffle(billList);
        for (int bb : billList) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double dotprod = y[bb] + x[bb] * billThetas[bb].dotProduct(us[aa]);
                    double score = Math.exp(dotprod);
                    double prob = score / (1 + score);
                    gradX += (getVote(aa, bb) - prob) * billThetas[bb].dotProduct(us[aa]);
                    gradY += getVote(aa, bb) - prob;
                }
            }
            // prior
            gradX -= x[bb] / sigma;
            gradY -= y[bb] / sigma;

            // update
            x[bb] += bRate * gradX;
            y[bb] += bRate * gradY;
        }
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

    }

    private String getAssignmentString() {
        StringBuilder assignStr = new StringBuilder();
        for (int d = 0; d < D; d++) {
            assignStr.append(d).append("\n");
            assignStr.append(DirMult.output(docTopics[d])).append("\n");
            for (int kk = 0; kk < K; kk++) {
                assignStr.append(DirMult.output(docFrames[d][kk])).append("\n");
            }
            for (int n = 0; n < topicZs[d].length; n++) {
                assignStr.append(topicZs[d][n]).append("\t");
                assignStr.append(frameZs[d][n]).append("\t");
            }
            assignStr.append("\n");
        }
        return assignStr.toString();
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
                modelStr.append(DirMult.output(topicWords[kk])).append("\n");
                for (int jj = 0; jj < J; jj++) {
                    modelStr.append(jj).append("\n");
                    modelStr.append(DirMult.output(frameWords[kk][jj])).append("\n");
                    modelStr.append(etas[kk][jj]).append("\n");
                }
            }

            // train author scores
            StringBuilder authorStr = new StringBuilder();
            authorStr.append(A).append("\n");
            for (int aa = 0; aa < A; aa++) {
                authorStr.append(MiscUtils.arrayToString(us[aa])).append("\n");
            }
            // train bill scores
            StringBuilder billStr = new StringBuilder();
            billStr.append(B).append("\n");
            for (int bb = 0; bb < B; bb++) {
                billStr.append(x[bb]).append("\n");
                billStr.append(y[bb]).append("\n");
            }

            // output to a compressed file
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(getAssignmentString());
            contentStrs.add(billStr.toString());
            contentStrs.add(authorStr.toString());

            String filename = IOUtils.removeExtension(IOUtils.getFilename(filepath));
            ArrayList<String> entryFiles = new ArrayList<>();
            entryFiles.add(filename + ModelFileExt);
            entryFiles.add(filename + AssignmentFileExt);
            entryFiles.add(filename + BillFileExt);
            entryFiles.add(filename + AuthorFileExt);

            this.outputZipFile(filepath, contentStrs, entryFiles);
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
            inputBillIdealPoints(filepath);
            inputAuthorIdealPoints(filepath);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Excepion while inputing state from "
                    + filepath);
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
            frameWords = new DirMult[K][J];
            etas = new double[K][J];
            for (int kk = 0; kk < K; kk++) {
                int topicIdx = Integer.parseInt(reader.readLine());
                if (topicIdx != kk) {
                    throw new MismatchRuntimeException(topicIdx, kk);
                }
                topicWords[kk] = DirMult.input(reader.readLine());
                for (int jj = 0; jj < J; jj++) {
                    int frameIdx = Integer.parseInt(reader.readLine());
                    if (frameIdx != jj) {
                        throw new MismatchRuntimeException(frameIdx, jj);
                    }
                    frameWords[kk][jj] = DirMult.input(reader.readLine());
                    etas[kk][jj] = Double.parseDouble(reader.readLine());
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing model from "
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
                int aa = authors[d];
                docTopics[d] = DirMult.input(reader.readLine());
                for (int kk = 0; kk < K; kk++) {
                    docFrames[d][kk] = DirMult.input(reader.readLine());
                }

                String[] sline = reader.readLine().split("\t");
                if (sline.length != words[d].length * 2) {
                    throw new RuntimeException("[MISMATCH]. Doc "
                            + d + ". " + sline.length + " vs. " + words[d].length * 2);
                }
                for (int n = 0; n < words[d].length; n++) {
                    topicZs[d][n] = Integer.parseInt(sline[2 * n]);
                    frameZs[d][n] = Integer.parseInt(sline[2 * n + 1]);
                    topicCounts[aa].increment(topicZs[d][n]);
                    frameCounts[aa][topicZs[d][n]].increment(frameZs[d][n]);
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
    }

    /**
     * Input author ideal points.
     *
     * @param zipFilepath File path
     */
    @Override
    public void inputAuthorIdealPoints(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + AuthorFileExt);
            int numAuthors = Integer.parseInt(reader.readLine());
            if (numAuthors != A) {
                throw new MismatchRuntimeException(numAuthors, A);
            }
            us = new double[A][];
            for (int aa = 0; aa < A; aa++) {
                us[aa] = MiscUtils.stringToDoubleArray(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    /**
     * Input bill ideal points.
     *
     * @param zipFilepath File path
     */
    @Override
    public void inputBillIdealPoints(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading bill scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + BillFileExt);

            int numBills = Integer.parseInt(reader.readLine());
            if (numBills != B) {
                throw new MismatchRuntimeException(numBills, B);
            }
            x = new double[B];
            y = new double[B];
            for (int bb = 0; bb < B; bb++) {
                x[bb] = Double.parseDouble(reader.readLine());
                y[bb] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
        }
    }

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
        this.outputTopicTopWords(file, numTopWords, topicVocab);
    }

    public void outputTopicTopWords(File file, int numTopWords, ArrayList<String> topicLabels) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + file);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            for (int kk = 0; kk < K; kk++) {
                String topicLabel = "Topic " + kk;
                if (topicLabels != null) {
                    topicLabel += ": " + topicLabels.get(kk);
                }

                double[] distrs = topicWords[kk].getDistribution();
                String[] topWords = getTopWords(distrs, numTopWords);
                writer.write("[" + topicLabel
                        + ", " + topicWords[kk].getCountSum() + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");

                // frames
                ArrayList<RankingItem<Integer>> sortedFrames = new ArrayList<RankingItem<Integer>>();
                for (int j = 0; j < J; j++) {
                    sortedFrames.add(new RankingItem<Integer>(j, etas[kk][j]));
                }
                Collections.sort(sortedFrames);

                for (int ii = 0; ii < J; ii++) {
                    int jj = sortedFrames.get(ii).getObject();
                    String frameLabel = "Frame " + kk + ":" + jj;
                    distrs = frameWords[kk][jj].getDistribution();
                    topWords = getTopWords(distrs, numTopWords);
                    writer.write("\t[" + frameLabel
                            + ", " + frameWords[kk][jj].getCountSum()
                            + ", " + MiscUtils.formatDouble(etas[kk][jj])
                            + "]");
                    for (String topWord : topWords) {
                        writer.write("\t" + topWord);
                    }
                    writer.write("\n\n");
                }
                writer.write("\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
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

        setTestConfigurations(); // set up test
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

    public void loadEtaList() {
        try {
            File reportFolder = new File(getReportFolderPath());
            String[] filenames = reportFolder.list();
            this.etaList = new ArrayList<>();
            for (String filename : filenames) {
                if (!filename.endsWith(".zip")) {
                    continue;
                }
                inputModel(new File(reportFolder, filename).getAbsolutePath());
                this.etaList.add(cloneEtas());
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading eta list");
        }
    }

    private double[][] cloneEtas() {
        double[][] cloneEtas = new double[K][J];
        for (int kk = 0; kk < K; kk++) {
            cloneEtas[kk] = etas[kk].clone();
        }
        return cloneEtas;
    }

    public void outputTheta(File outputFile) {
        if (verbose) {
            logln("Output thetas to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Topic\tCount\n");
            for (int kk = 0; kk < K; kk++) {
                writer.write(getTopicLabel(kk) + "\t" + topicWords[kk].getCountSum() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    public void outputHierarchyWithDetails(File outputFile,
            HashMap<String, Author> authorTable,
            String[] docIds,
            String[][] docRawSentences) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }

        if (verbose) {
            logln("Outputing per-topic top words to " + outputFile);
        }

        // validate
        for (int aa = 0; aa < A; aa++) {
            int authorTokens = 0;
            for (int kk = 0; kk < K; kk++) {
                authorTokens += frameCounts[aa][kk].getCountSum();
            }
            if (authorTokens != authorTotalWordWeights[aa]) {
                throw new MismatchRuntimeException(authorTokens, (int) authorTotalWordWeights[aa]);
            }
        }

        if (this.authorVocab.size() != A) {
            throw new MismatchRuntimeException(this.authorVocab.size(), A);
        }

        ArrayList<Integer>[] authorDocIndices = new ArrayList[A];
        for (int aa = 0; aa < A; aa++) {
            authorDocIndices[aa] = new ArrayList<>();
        }
        for (int dd = 0; dd < D; dd++) {
            authorDocIndices[authors[dd]].add(dd);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            BufferedWriter textWriter = IOUtils.getBufferedWriter(outputFile + ".text");

            for (int kk = 0; kk < K; kk++) {
                double[] distrs = topicWords[kk].getDistribution();
                String[] topWords = getTopWords(distrs, 15);
                writer.write("[" + getTopicLabel(kk)
                        + ", " + topicWords[kk].getCountSum() + "]");
                for (String topWord : topWords) {
                    writer.write("\t" + topWord);
                }
                writer.write("\n\n");

                // frames
                ArrayList<RankingItem<Integer>> sortedFrames = new ArrayList<RankingItem<Integer>>();
                for (int j = 0; j < J; j++) {
                    sortedFrames.add(new RankingItem<Integer>(j, etas[kk][j]));
                }
                Collections.sort(sortedFrames);

                for (int ii = 0; ii < J; ii++) {
                    int jj = sortedFrames.get(ii).getObject();
                    ArrayList<Double> frameEtas = new ArrayList<>();
                    for (double[][] e : etaList) {
                        frameEtas.add(e[kk][jj]);
                    }

                    ArrayList<RankingItem<Integer>> rankAuthors = new ArrayList<>();
                    for (int aa = 0; aa < A; aa++) {
                        double val = 0.0;
                        if (authorTotalWordWeights[aa] != 0.0) {
                            val = (double) frameCounts[aa][kk].getCount(jj) / authorTotalWordWeights[aa];
                        }
                        rankAuthors.add(new RankingItem<Integer>(aa, val));
                    }
                    Collections.sort(rankAuthors);

                    String frameLabel = "Frame " + kk + ":" + jj;
                    distrs = frameWords[kk][jj].getDistribution();
                    topWords = getTopWords(distrs, 15);
                    writer.write("\t[" + frameLabel
                            + ", " + frameWords[kk][jj].getCountSum()
                            + ", " + MiscUtils.formatDouble(etas[kk][jj])
                            + ", " + MiscUtils.formatDouble(StatUtils.mean(frameEtas))
                            + ", " + MiscUtils.formatDouble(StatUtils.standardDeviation(frameEtas))
                            + "]");
                    for (String topWord : topWords) {
                        writer.write("\t" + topWord);
                    }

                    writer.write("\n");
                    for (int xx = 0; xx < 10; xx++) {
                        RankingItem<Integer> rankAuthor = rankAuthors.get(xx);
                        int aa = rankAuthor.getObject();
                        String authorId = authorVocab.get(aa);
                        Author author = authorTable.get(authorId);

                        writer.write("\t\t" + aa
                                + "\t" + authorId
                                + "\t" + MiscUtils.formatDouble(rankAuthor.getPrimaryValue())
                                + "\t" + authorTotalWordWeights[aa]
                                + "\t" + us[aa][kk]
                                + "\t" + author.getProperty(GTLegislator.NAME)
                                + "\t" + author.getProperty(GTLegislator.NOMINATE_SCORE1)
                                + "\n");

                        // rank document
                        ArrayList<RankingItem<Integer>> rankDocs = new ArrayList<>();
                        for (int dd : authorDocIndices[aa]) {
                            int count = docFrames[dd][kk].getCount(jj);
                            if (count > 0) {
                                rankDocs.add(new RankingItem<Integer>(dd, (double) count));
                            }
                        }
                        Collections.sort(rankDocs);

                        for (int yy = 0; yy < Math.min(5, rankDocs.size()); yy++) {
                            RankingItem<Integer> rankDoc = rankDocs.get(yy);
                            int dd = rankDoc.getObject();
                            writer.write("\t\t\t" + dd
                                    + "\t" + docIds[docIndices.get(dd)]
                                    + ". " + rankDoc.getPrimaryValue()
                                    + ". " + words[dd].length
                                    + "\n");

                            StringBuilder docStr = new StringBuilder();
                            for (String ss : docRawSentences[docIndices.get(dd)]) {
                                docStr.append(ss).append(" ");
                            }
                            textWriter.write(kk + ", " + jj
                                    + ", " + dd
                                    + "\t" + docIds[docIndices.get(dd)]
                                    + "\t" + words[dd].length
                                    + "\t" + docStr.toString() + "\n\n");
                        }
                    }

                    writer.write("\n\n");
                }
                writer.write("\n\n");
            }
            writer.close();
            textWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
        }
    }

    public void outputTopicFrameVariance(File outputFile) {
        if (verbose) {
            logln("Output variance to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Topic");
            for (int jj = 0; jj < J; jj++) {
                writer.write("\tFrame-" + jj);
            }
            writer.write("\tMax\tMin\tAvg\tMedian\tWeightAvg\n");

            for (int kk = 0; kk < K; kk++) {
                writer.write(getTopicLabel(kk));

                double num = 0.0;
                double den = 0.0;
                for (int jj = 0; jj < J; jj++) {
                    writer.write("\t" + etas[kk][jj]);

                    num += etas[kk][jj] * frameWords[kk][jj].getCountSum();
                    den += frameWords[kk][jj].getCountSum();
                }
                writer.write("\t" + StatUtils.max(etas[kk]));
                writer.write("\t" + StatUtils.min(etas[kk]));
                writer.write("\t" + StatUtils.mean(etas[kk]));
                writer.write("\t" + StatUtils.median(etas[kk]));
                writer.write("\t" + (num / den));
                writer.write("\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    private String getTopicLabel(int kk) {
        return "\"" + this.topicVocab.get(kk) + "\"";
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

            if (iter % testSampleLag == 0) {
                SparseVector[] predictions = predictOutMatrix();

                if (iter >= testBurnIn) { // store partial prediction
                    predictionList.add(predictions);
                }

                if (votes != null) { // for debug
                    ArrayList<Measurement> measurements = AbstractVotePredictor
                            .evaluate(votes, validVotes, predictions);
                    for (Measurement m : measurements) {
                        logln(">>> o >>> " + m.getName() + ": " + m.getValue());
                    }
                }
            }
        }

        if (assignmentFile != null) { // output assignments of test data
            try { // output to a compressed file
                ArrayList<String> contentStrs = new ArrayList<>();
                contentStrs.add(getAssignmentString());

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
            HierMultiTIPM sampler) {
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
                HierMultiTIPMRunner runner = new HierMultiTIPMRunner(
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

class HierMultiTIPMRunner implements Runnable {

    HierMultiTIPM sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public HierMultiTIPMRunner(HierMultiTIPM sampler,
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
        HierMultiTIPM testSampler = new HierMultiTIPM();
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
            testSampler.test(stateFile, predictionFile, null);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }
}
