package votepredictor.textidealpoint.flat;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Author;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import optimization.OWLQNLinearRegression;
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
import util.evaluation.MimnoTopicCoherence;
import util.govtrack.GTLegislator;
import votepredictor.AbstractVotePredictor;
import votepredictor.textidealpoint.AbstractTextSingleIdealPoint;

/**
 *
 * @author vietan
 */
public class HybridSLDAIdealPoint extends AbstractTextSingleIdealPoint {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double gamma; // eta variance
    public double lambda; // l2-norm using L-BFGS
    public double l1; // l1-norm using OWL-QN
    public double l2; // l2-norm using OWL-QN

    // input
    protected int K; // number of topics
    protected double[][] priorTopics;

    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;

    protected double[] eta; // regression parameters for topics
    protected SparseVector[] tau; // each topic has its own lexical regression
    protected double[] topVals;
    protected double[] lexVals;

    private ArrayList<String> labelVocab;

    public HybridSLDAIdealPoint() {
        this.basename = "Hybrid-SLDA-ideal-point";
    }

    public HybridSLDAIdealPoint(String bname) {
        this.basename = bname;
    }

    public void setLabelVocab(ArrayList<String> labelVoc) {
        this.labelVocab = labelVoc;
    }

    public void configure(HybridSLDAIdealPoint sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.rho,
                sampler.sigma,
                sampler.gamma,
                sampler.lambda,
                sampler.l1,
                sampler.l2,
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
            int V, int K,
            double alpha,
            double beta,
            double rho,
            double sigma, // stadard deviation of Gaussian for regression parameters
            double gamma,
            double lambda,
            double l1, double l2,
            double[][] priorTopics,
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
        this.lambda = lambda;
        this.l1 = l1;
        this.l2 = l2;
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

        boolean isLbfgs = this.lambda > 0;

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append("_B-").append(BURN_IN)
                .append("_M-").append(MAX_ITER)
                .append("_L-").append(LAG)
                .append("_K-").append(K)
                .append("_a-").append(formatter.format(hyperparams.get(ALPHA)))
                .append("_b-").append(formatter.format(hyperparams.get(BETA)))
                .append("_r-").append(formatter.format(this.rho))
                .append("_s-").append(formatter.format(this.sigma))
                .append("_g-").append(formatter.format(this.gamma))
                .append(isLbfgs ? ("_l-" + MiscUtils.formatDouble(this.lambda, 10)) : "")
                .append(!isLbfgs ? ("_l1-" + MiscUtils.formatDouble(this.l1, 10)) : "")
                .append(!isLbfgs ? ("_l2-" + MiscUtils.formatDouble(this.l2, 10)) : "");
        str.append("_opt-").append(this.paramOptimized);
        str.append("_prior-").append(this.priorTopics != null);
        this.name = str.toString();

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- num word types:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- gamma:\t" + MiscUtils.formatDouble(gamma));
            logln("--- lambda:\t" + MiscUtils.formatDouble(lambda));
            logln("--- lambda:\t" + MiscUtils.formatDouble(lambda));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1, 10));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2, 10));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- has prior? " + (priorTopics != null));
        }
    }

    public double[] getPredictedUs() {
        double[] predUs = new double[A];
        for (int aa = 0; aa < A; aa++) {
            predUs[aa] = topVals[aa] + lexVals[aa];
        }
        return predUs;
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

        initializeIdealPoint(); // initialize ideal points
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
        tau = new SparseVector[K];
        for (int kk = 0; kk < K; kk++) {
            tau[kk] = new SparseVector(V);
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
        topVals = new double[A];
        lexVals = new double[A];
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
                throw new RuntimeException("Initialization " + initState + " not supported");
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

        LDA lda = this.runLDA(K);
        int[][] ldaZ = lda.getZs();

        // initialize assignments
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = ldaZ[dd][nn];
                docTopics[dd].increment(z[dd][nn]);
                topicWords[z[dd][nn]].increment(words[dd][nn]);
                topVals[aa] += eta[z[dd][nn]] / authorTotalWordWeights[aa];
                lexVals[aa] += tau[z[dd][nn]].get(words[dd][nn]) / authorTotalWordWeights[aa];
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
                    double score = Math.exp((topVals[aa] + lexVals[aa]) * x[bb] + y[bb]);
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
            double diff = u[aa] - (topVals[aa] + lexVals[aa]);
            mse += diff * diff;
        }
        return mse / A;
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
        ArrayList<Measurement> measurements = AbstractVotePredictor
                .evaluate(votes, validVotes, predictions);
        for (Measurement m : measurements) {
            logln(">>> i >>> " + m.getName() + ": " + m.getValue());
        }

        predictions = predictOutMatrix();
        measurements = AbstractVotePredictor.evaluate(votes, validVotes, predictions);
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
        for (int d = 0; d < D; d++) {
            int aa = authors[d];
            for (int n = 0; n < words[d].length; n++) {
                if (removeFromModel) {
                    topicWords[z[d][n]].decrement(words[d][n]);
                }
                if (removeFromData) {
                    docTopics[d].decrement(z[d][n]);
                    topVals[aa] -= eta[z[d][n]] / authorTotalWordWeights[aa];
                    lexVals[aa] -= tau[z[d][n]].get(words[d][n]) / authorTotalWordWeights[aa];
                }

                double[] logprobs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    logprobs[kk]
                            = Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[kk].getProbability(words[d][n]));
                    if (observe) {
                        double aMean = topVals[aa] + lexVals[aa]
                                + (eta[kk] + tau[kk].get(words[d][n])) / authorTotalWordWeights[aa];
                        double resLLh = StatUtils.logNormalProbability(u[aa], aMean, Math.sqrt(rho));
                        logprobs[kk] += resLLh;
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
                    topVals[aa] += eta[z[d][n]] / authorTotalWordWeights[aa];
                    lexVals[aa] += tau[z[d][n]].get(words[d][n]) / authorTotalWordWeights[aa];
                }
            }
        }
        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
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

        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(K);
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int kk : docTopics[dd].getSparseCounts().getIndices()) {
                int count = docTopics[dd].getCount(kk);
                designMatrix[aa].change(kk, (double) count / authorTotalWordWeights[aa]);
            }
        }

        double[] responses = new double[A];
        for (int aa = 0; aa < A; aa++) {
            responses[aa] = u[aa] - lexVals[aa];
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, eta, designMatrix, rho, 0.0, gamma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update regression parameters
        for (int kk = 0; kk < K; kk++) {
            eta[kk] = optimizable.getParameter(kk);
        }

        for (int aa = 0; aa < A; aa++) {
            topVals[aa] = designMatrix[aa].dotProduct(eta);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
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
                docTopicWordCounts[dd][z[dd][nn]].increment(words[dd][nn]);
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
                    lexVals[aa] -= count * tau[kk].get(vv) / authorTotalWordWeights[aa];
                }
            }

            for (int aa = 0; aa < A; aa++) {
                lexResponses[aa] = u[aa] - topVals[aa] - lexVals[aa];
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
                lexVals[aa] += lexDesginMatrix[aa].dotProduct(tau[kk]);
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
                docTopicWordCounts[dd][z[dd][nn]].increment(words[dd][nn]);
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
//                    lexicalVals[aa] -= count * tau[kk].get(vv) / authorTotalWordWeights[aa];
                }
            }

            for (int aa = 0; aa < A; aa++) {
                lexVals[aa] -= lexDesginMatrix[aa].dotProduct(tau[kk]);
                lexResponses[aa] = u[aa] - topVals[aa] - lexVals[aa];
            }

//            for (int aa = 0; aa < A; aa++) {
//                lexResponses[aa] = u[aa] - topicVals[aa] - lexicalVals[aa];
//            }
            OWLQNLinearRegression opt = new OWLQNLinearRegression(basename, l1, l2);
            opt.setQuiet(true);
            OWLQNLinearRegression.setVerbose(false);
            double[] topicTau = tau[kk].dense();
            try {
                opt.train(lexDesginMatrix, lexResponses, topicTau);
            } catch (Exception e) {
                e.printStackTrace();
            }
            tau[kk] = new SparseVector(topicTau);

            // update
            for (int aa = 0; aa < A; aa++) {
                lexVals[aa] += lexDesginMatrix[aa].dotProduct(tau[kk]);
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
            grad -= (u[aa] - topVals[aa] - lexVals[aa]) / rho; // prior
            u[aa] += aRate * grad; // update
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
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(vv).append("\t").append(this.wordWeights[vv]).append("\n");
            }
            for (int kk = 0; kk < K; kk++) {
                modelStr.append(kk).append("\n");
                modelStr.append(eta[kk]).append("\n");
                modelStr.append(DirMult.output(topicWords[kk])).append("\n");
            }
            for (int kk = 0; kk < K; kk++) {
                modelStr.append(SparseVector.output(tau[kk])).append("\n");
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
            ArrayList<String> contentStrs = new ArrayList<>();
            contentStrs.add(modelStr.toString());
            contentStrs.add(assignStr.toString());
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

        validate("Done reading state from " + filepath);
    }

    public void inputModel(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading model from " + zipFilepath);
        }

        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath,
                    filename + ModelFileExt);
            wordWeights = new double[V];
            for (int vv = 0; vv < V; vv++) {
                String[] sline = reader.readLine().split("\t");
                int vIdx = Integer.parseInt(sline[0]);
                if (vv != vIdx) {
                    throw new MismatchRuntimeException(vIdx, vv);
                }
                wordWeights[vv] = Double.parseDouble(sline[1]);
            }

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

            tau = new SparseVector[K];
            for (int kk = 0; kk < K; kk++) {
                tau[kk] = SparseVector.input(reader.readLine());
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

    @Override
    public void outputTopicTopWords(File file, int numTopWords) {
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
                if (labelVocab != null) {
                    topicLabel = labelVocab.get(k);
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

                writer.write("\n");

                int nonzero = 0;
                ArrayList<RankingItem<Integer>> rankLexs = new ArrayList<>();
                for (int vv : tau[k].getIndices()) {
                    rankLexs.add(new RankingItem<Integer>(vv, tau[k].get(vv)));
                    nonzero++;
                }
                Collections.sort(rankLexs);
                for (int jj = 0; jj < Math.min(10, rankLexs.size()); jj++) {
                    RankingItem<Integer> rankLex = rankLexs.get(jj);
                    if (rankLex.getPrimaryValue() > 0) {
                        writer.write("+++ " + wordVocab.get(rankLex.getObject())
                                + "\t" + MiscUtils.formatDouble(rankLex.getPrimaryValue())
                                + ", " + topicWords[k].getCount(rankLex.getObject())
                                + "\n");
                    }
                }

                for (int jj = 0; jj < Math.min(10, rankLexs.size()); jj++) {
                    RankingItem<Integer> rankLex = rankLexs.get(rankLexs.size() - 1 - jj);
                    if (rankLex.getPrimaryValue() < 0) {
                        writer.write("--- " + wordVocab.get(rankLex.getObject())
                                + "\t" + MiscUtils.formatDouble(rankLex.getPrimaryValue())
                                + ", " + topicWords[k].getCount(rankLex.getObject())
                                + "\n");
                    }
                }
                int numObs = topicWords[k].getSparseCounts().getIndices().size();
                writer.write(">>> # non-zero count: " + numObs + "\n");
                writer.write(">>> # non-zero tau: " + nonzero + "\n\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing topic words");
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
                writer.write(authorName + "\t" + authorFWID + "\t" + u[aa]
                        + "\t" + topVals[aa] + "\t" + lexVals[aa]
                        + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + authorAnalysisFile);
        }
    }

    /**
     * Compute the averaged topic coherence
     *
     * @param file Output result file
     * @param topicCoherence Prepared Mimno's topic coherence
     * @return
     */
    public double[][] computeAvgTopicCoherence(File file,
            MimnoTopicCoherence topicCoherence) {
        if (this.wordVocab == null) {
            throw new RuntimeException("The word vocab has not been assigned yet");
        }
        if (verbose) {
            logln("Evaluating coherence ...");
        }

        File reportFolder = new File(getSamplerFolderPath(), ReportFolder);
        if (!reportFolder.exists()) {
            throw new RuntimeException("Report folder does not exist. " + reportFolder);
        }
        String[] filenames = reportFolder.list();
        double[][] avgTopics = new double[K][V];
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file.getAbsolutePath() + ".iter");
            writer.write("Iteration");
            for (int k = 0; k < K; k++) {
                writer.write("\tTopic_" + k);
            }
            writer.write("\n");

            // partial score
            ArrayList<double[][]> aggTopics = new ArrayList<double[][]>();
            for (String filename : filenames) {
                if (!filename.contains("zip")) {
                    continue;
                }
                inputModel(new File(reportFolder, filename).getAbsolutePath());
                double[][] pointTopics = new double[K][V];

                writer.write(filename);
                for (int k = 0; k < K; k++) {
                    pointTopics[k] = topicWords[k].getDistribution();
                    int[] topic = SamplerUtils.getSortedTopic(pointTopics[k]);
                    double score = topicCoherence.getCoherenceScore(topic);

                    writer.write("\t" + score);
                }
                writer.write("\n");
                aggTopics.add(pointTopics);
            }

            // averaging
            writer.write("Average");
            ArrayList<Double> scores = new ArrayList<Double>();
            for (int k = 0; k < K; k++) {
                double[] avgTopic = new double[V];
                for (int v = 0; v < V; v++) {
                    for (double[][] aggTopic : aggTopics) {
                        avgTopic[v] += aggTopic[k][v] / aggTopics.size();
                    }
                }
                int[] topic = SamplerUtils.getSortedTopic(avgTopic);
                double score = topicCoherence.getCoherenceScore(topic);
                writer.write("\t" + score);
                scores.add(score);
                avgTopics[k] = avgTopic;
            }
            writer.write("\n");
            writer.close();

            // output aggregated topic coherence scores
            if (verbose) {
                logln("Outputing averaged topic coherence to file " + file);
            }
            IOUtils.outputTopicCoherences(file, scores);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while sampling during test time.");
        }
        return avgTopics;
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

        // debug
//        System.out.println("A = " + A + ". " + authorTotalWordWeights.length);
//        System.out.println("sum lex: " + StatUtils.sum(lexVals));
//        System.out.println("sum top: " + StatUtils.sum(topVals));
//        System.exit(1);
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
            HybridSLDAIdealPoint sampler) {
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
                HybridSLDATestRunner runner = new HybridSLDATestRunner(
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

class HybridSLDATestRunner implements Runnable {

    HybridSLDAIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public HybridSLDATestRunner(HybridSLDAIdealPoint sampler,
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
        HybridSLDAIdealPoint testSampler = new HybridSLDAIdealPoint();
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
