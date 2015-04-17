package votepredictor.textidealpoint.flat;

import cc.mallet.optimize.LimitedMemoryBFGS;
import edu.stanford.nlp.optimization.DiffFunction;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
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
import votepredictor.AbstractVotePredictor;
import votepredictor.textidealpoint.AbstractTextSingleIdealPoint;

/**
 *
 * @author vietan
 */
public class LexicalSLDAIdealPoint extends AbstractTextSingleIdealPoint {

    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double gamma;    // eta variance
    public double lambda;   // l2-norm using L-BFGS
    public double l1;       // l1-norm using OWL-QN
    public double l2;       // l2-norm using OWL-QN
    // input
    protected int K; // number of topics
    protected double[][] priorTopics;
    // derived
    protected SparseVector[] wa;
    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;
    protected double[] eta; // regression parameters for topics
    protected double[] tau; // lexical regression parameters
    // internal
    protected double[] zaEta;
    protected double[] waTau;
    protected boolean isLexReg;
    protected boolean isTopReg;

    public LexicalSLDAIdealPoint() {
        this.basename = "Lexical-SLDA-ideal-point";
    }

    public LexicalSLDAIdealPoint(String bname) {
        this.basename = bname;
    }

    public void configure(LexicalSLDAIdealPoint sampler) {
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
            double sigma,
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

        this.isTopReg = this.K > 0;
        this.isLexReg = this.lambda > 0 || this.l1 > 0 || this.l2 > 0;

        StringBuilder str = new StringBuilder();
        str.append(this.prefix)
                .append("_").append(basename)
                .append(isTopReg ? ("_B-" + BURN_IN) : "")
                .append(isTopReg ? ("_M-" + MAX_ITER) : "")
                .append(isTopReg ? ("_L-" + LAG) : "")
                .append(isTopReg ? ("_K-" + K) : "")
                .append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma)) // ideal point var
                .append(isTopReg ? ("_a-" + formatter.format(hyperparams.get(ALPHA))) : "")
                .append(isTopReg ? ("_b-" + formatter.format(hyperparams.get(BETA))) : "")
                .append(isTopReg ? ("_g-" + formatter.format(gamma)) : "")
                .append("_l-").append(formatter.format(lambda)) // tau var
                .append("_l1-").append(MiscUtils.formatDouble(l1, 10))
                .append("_l2-").append(MiscUtils.formatDouble(l2, 10));
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
            logln("--- lambda:\t" + MiscUtils.formatDouble(lambda, 10));
            logln("--- l1:\t" + MiscUtils.formatDouble(l1, 10));
            logln("--- l2:\t" + MiscUtils.formatDouble(l2, 10));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- isLexReg? " + isLexReg);
            logln("--- isTopReg? " + isTopReg);
            logln("--- has prior? " + (priorTopics != null));
        }
    }

    public double[] getPredictedUs() {
        double[] predUs = new double[A];
        for (int aa = 0; aa < A; aa++) {
            predUs[aa] = zaEta[aa] + waTau[aa];
        }
        return predUs;
    }

    @Override
    protected void prepareDataStatistics() {
        super.prepareDataStatistics();

        this.wa = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.wa[aa] = new SparseVector(V);
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            SparseCount counts = new SparseCount();
            for (int nn = 0; nn < words[dd].length; nn++) {
                counts.increment(words[dd][nn]);
            }

            for (int vv : counts.getIndices()) {
                int count = counts.getCount(vv);
                this.wa[aa].change(vv, count);
            }
        }

        for (int aa = 0; aa < A; aa++) {
            this.wa[aa].normalize();
        }
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

        tau = new double[V];
        waTau = new double[A];
        if (isLexReg) {
            initLexicalRegModel();
            initLexicalRegData();
        }

        eta = new double[K];
        zaEta = new double[A];
        if (isTopReg) {
            initTopicRegModel(seededTopics);
            initTopicRegData();
            initializeAssignments();
        }

        if (debug) {
            validate("Initialized");
        }

        if (verbose) {
            logln("--- Done initializing. \n" + getCurrentState());
        }
    }

    protected void initLexicalRegModel() {
        for (int vv = 0; vv < V; vv++) {
            this.tau[vv] = SamplerUtils.getGaussian(0.0, gamma);
        }
    }

    protected void initLexicalRegData() {
        for (int aa = 0; aa < A; aa++) {
            waTau[aa] = wa[aa].dotProduct(tau);
        }
    }

    protected void initTopicRegModel(double[][] seededTopics) {
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

        for (int kk = 0; kk < K; kk++) {
            eta[kk] = SamplerUtils.getGaussian(0.0, gamma);
        }
    }

    protected void initTopicRegData() {
        docTopics = new DirMult[D];
        for (int d = 0; d < D; d++) {
            docTopics[d] = new DirMult(K, hyperparams.get(ALPHA) * K, 1.0 / K);
        }

        z = new int[D][];
        for (int d = 0; d < D; d++) {
            z[d] = new int[words[d].length];
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

        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                z[dd][nn] = ldaZ[dd][nn];
                docTopics[dd].increment(z[dd][nn]);
                topicWords[z[dd][nn]].increment(words[dd][nn]);
                zaEta[aa] += eta[z[dd][nn]] / authorTotalWordWeights[aa];
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
                    double score = Math.exp((zaEta[aa] + waTau[aa]) * x[bb] + y[bb]);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
    }

    /**
     * Compute MSE.
     *
     * @return
     */
    protected double getMSE() {
        double mse = 0.0;
        for (int aa = 0; aa < A; aa++) {
            double diff = u[aa] - (zaEta[aa] + waTau[aa]);
            mse += diff * diff;
        }
        return mse / A;
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        System.out.println();
        this.getLogLikelihood();
        this.evaluate();
        return str.toString();
    }

    private void evaluate() {
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
    }

    @Override
    public void metaIterate() {
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

        if (isLexReg) {
            if (lambda > 0) {
                updateTausLBFGS();
            } else {
                updateTausOWLQN();
            }
        }

        if (isTopReg) { // only sample if doing topic regression
            for (iter = 0; iter < MAX_ITER; iter++) {
                isReporting = isReporting();
                if (isReporting) {
                    if (iter < BURN_IN) {
                        logln("--- Burning in. Iter " + iter + "/" + MAX_ITER
                                + " @ " + Thread.currentThread().getId());
                    } else {
                        logln("--- Sampling. Iter " + iter + "/" + MAX_ITER
                                + " @ " + Thread.currentThread().getId());
                    }
                    logln(getCurrentState());
                }

                iterate();

                // store model
                if (report && iter > BURN_IN && iter % LAG == 0) {
                    outputState(new File(reportFolderPath, getIteratedStateFile()));
                    outputTopicTopWords(new File(reportFolderPath, getIteratedTopicFile()), 15);
                }
            }
        }

        if (report) { // output the final model
            outputState(new File(reportFolderPath, getIteratedStateFile()));
            outputTopicTopWords(new File(reportFolderPath, getIteratedTopicFile()), 15);
        }

        if (debug) {
            validate("iter " + iter);
        }

        float ellapsedSeconds = (System.currentTimeMillis() - startTime) / (1000);
        logln("Total runtime iterating: " + ellapsedSeconds + " seconds");

        if (log && isLogging()) {
            closeLogger();
        }
    }

    @Override
    public void iterate() {
        if (isTopReg) {
            updateEtas(); // update topic regression parameters
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED); // sample token assignments
        }

        // update lexical regression parameters
        if (isLexReg) {
            if (lambda > 0) {
                updateTausLBFGS();
            } else {
                updateTausOWLQN();
            }
        }

        // update ideal points
        if (iter >= BURN_IN) {
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
                    zaEta[aa] -= eta[z[d][n]] / authorTotalWordWeights[aa];
                }

                double[] logprobs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    logprobs[kk]
                            = Math.log(docTopics[d].getCount(kk) + hyperparams.get(ALPHA))
                            + Math.log(topicWords[kk].getProbability(words[d][n]));
                    if (observe) {
                        double aMean = waTau[aa] + zaEta[aa]
                                + eta[kk] / authorTotalWordWeights[aa];
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
                    zaEta[aa] += eta[z[d][n]] / authorTotalWordWeights[aa];
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
            responses[aa] = u[aa] - waTau[aa];
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

        // update dot products
        for (int aa = 0; aa < A; aa++) {
            zaEta[aa] = designMatrix[aa].dotProduct(eta);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    public long updateTausLBFGS() {
        if (isReporting) {
            logln("+++ Updating tau ...");
        }
        long sTime = System.currentTimeMillis();

        double[] responses = new double[A];
        for (int aa = 0; aa < A; aa++) {
            responses[aa] = u[aa] - zaEta[aa];
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                responses, tau, wa, rho, 0.0, lambda);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        for (int vv = 0; vv < V; vv++) { // update regression parameters
            tau[vv] = optimizable.getParameter(vv);
        }

        for (int aa = 0; aa < A; aa++) { // update
            waTau[aa] = wa[aa].dotProduct(tau);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- converged? " + converged);
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    protected long updateTausOWLQN() {
        if (isReporting) {
            logln("+++ Updating tau using OWL-QN ...");
        }
        long sTime = System.currentTimeMillis();

        double[] responses = new double[A];
        for (int aa = 0; aa < A; aa++) {
            responses[aa] = u[aa] - zaEta[aa];
        }

        // optimize
        OWLQNLinearRegression opt = new OWLQNLinearRegression(basename, l1, l2);
        opt.setQuiet(true);
        OWLQNLinearRegression.setVerbose(false);
        opt.train(wa, responses, tau);

        for (int aa = 0; aa < A; aa++) { // update
            this.waTau[aa] = wa[aa].dotProduct(tau);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    public long updateEtasAndTaus() {
        if (isReporting) {
            logln("+++ Updating etas and taus...");
        }
        long sTime = System.currentTimeMillis();

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
            grad -= (u[aa] - zaEta[aa] - waTau[aa]) / rho; // prior
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
                modelStr.append(vv).append("\t").append(wordWeights[vv]).append("\n");
            }

            if (isTopReg) {
                for (int kk = 0; kk < K; kk++) {
                    modelStr.append(kk).append("\n");
                    modelStr.append(eta[kk]).append("\n");
                    modelStr.append(DirMult.output(topicWords[kk])).append("\n");
                }
            }

            if (isLexReg) {
                for (int vv = 0; vv < V; vv++) {
                    modelStr.append(vv).append("\t").append(tau[vv]).append("\n");
                }
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
            if (isTopReg) {
                for (int d = 0; d < D; d++) {
                    assignStr.append(d).append("\n");
                    assignStr.append(DirMult.output(docTopics[d])).append("\n");
                    for (int n = 0; n < z[d].length; n++) {
                        assignStr.append(z[d][n]).append("\t");
                    }
                    assignStr.append("\n");
                }
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
            if (isTopReg) {
                inputAssignments(filepath);
            }
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

            eta = new double[K];
            if (isTopReg) {
                topicWords = new DirMult[K];
                for (int kk = 0; kk < K; kk++) {
                    int topicIdx = Integer.parseInt(reader.readLine());
                    if (topicIdx != kk) {
                        throw new RuntimeException("Indices mismatch when loading model");
                    }
                    eta[kk] = Double.parseDouble(reader.readLine());
                    topicWords[kk] = DirMult.input(reader.readLine());
                }
            }

            tau = new double[V];
            if (isLexReg) {
                for (int vv = 0; vv < V; vv++) {
                    String[] sline = reader.readLine().split("\t");
                    int vIdx = Integer.parseInt(sline[0]);
                    if (vv != vIdx) {
                        throw new MismatchRuntimeException(vIdx, vv);
                    }
                    tau[vv] = Double.parseDouble(sline[1]);
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
            docTopics = new DirMult[D];
            z = new int[D][];
            for (int d = 0; d < D; d++) {
                z[d] = new int[words[d].length];
            }
            zaEta = new double[A];
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

                String[] sline = reader.readLine().split("\t");
                if (sline.length != words[d].length) {
                    throw new RuntimeException("[MISMATCH]. Doc "
                            + d + ". " + sline.length + " vs. " + words[d].length);
                }
                for (int n = 0; n < words[d].length; n++) {
                    z[d][n] = Integer.parseInt(sline[n]);
                    zaEta[aa] += eta[z[d][n]] / authorTotalWordWeights[aa];
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

    public void outputLexicalItems(File outputFile) {
        ArrayList<RankingItem<Integer>> rankWords = new ArrayList<>();
        for (int vv = 0; vv < V; vv++) {
            rankWords.add(new RankingItem<Integer>(vv, tau[vv]));
        }
        Collections.sort(rankWords);

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (int vv = 0; vv < V; vv++) {
                RankingItem<Integer> rankWord = rankWords.get(vv);
                if (rankWord.getPrimaryValue() != 0) {
                    writer.write(vv
                            + "\t" + rankWord.getObject()
                            + "\t" + wordVocab.get(rankWord.getObject())
                            + "\t" + rankWord.getPrimaryValue()
                            + "\n");
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing " + outputFile);
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

        SparseVector[] predictions;
        if (isTopReg) {
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
            predictions = new SparseVector[validVotes.length];
            for (int aa = 0; aa < A; aa++) {
                int author = authorIndices.get(aa);
                predictions[author] = new SparseVector(validVotes[author].length);
                for (SparseVector[] pred : predictionList) {
                    predictions[author].add(pred[author]);
                }
                predictions[author].scale(1.0 / predictionList.size());
            }

            if (predictionFile != null) { // output predictions
                AbstractVotePredictor.outputPredictions(predictionFile, null, predictions);
            }
        } else {
            assert isLexReg == true;
            initLexicalRegData();
            predictions = predictOutMatrix();
        }
        return predictions;
    }

    /**
     * Sample topic assignments for all tokens in a set of test documents. This
     * is done only if topic regression is performed.
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

        zaEta = new double[A];
        waTau = new double[A];
        initLexicalRegData();
        initTopicRegData();

        // sample
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
        for (iter = 0; iter < testMaxIter; iter++) {
            isReporting = verbose && iter % testRepInterval == 0;
            if (isReporting) {
                if (iter < testBurnIn) {
                    logln("--- Burning in. Iter " + iter + "/" + testMaxIter
                            + " @ " + Thread.currentThread().getId());
                } else {
                    logln("--- Sampling. Iter " + iter + "/" + testMaxIter
                            + " @ " + Thread.currentThread().getId());
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
            LexicalSLDAIdealPoint sampler) {
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
                LexicalSLDATestRunner runner = new LexicalSLDATestRunner(
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

    class DiffFunc implements DiffFunction {

        // inputs
        private final double[] values; // [A]-dim vector
        private final SparseVector[] designMatrix; // [A]x[K+V] sparse matrix
        // derived
        private final double topL;
        private final double lexL;

        public DiffFunc(SparseVector[] designMatrix, double[] values,
                double topL, double lexL) {
            this.designMatrix = designMatrix;
            this.values = values;
            this.topL = topL;
            this.lexL = lexL;
        }

        @Override
        public int domainDimension() {
            return K + V;
        }

        @Override
        public double valueAt(double[] w) {
            double loss = 0.0;
            for (int aa = 0; aa < A; aa++) {
                double dotprod = designMatrix[aa].dotProduct(w);
                double diff = values[aa] - dotprod;
                loss += diff * diff;
            }

            double reg = 0.0;
            for (int kk = 0; kk < K; kk++) {
                reg += topL * w[kk] * w[kk];
            }

            for (int vv = 0; vv < V; vv++) {
                reg += lexL * w[K + vv] * w[K + vv];
            }

            return loss + reg;
        }

        @Override
        public double[] derivativeAt(double[] w) {
            double[] grads = new double[K + V];
            for (int aa = 0; aa < A; aa++) {
                double dotprod = designMatrix[aa].dotProduct(w);
                for (int k : this.designMatrix[aa].getIndices()) {
                    grads[k] -= 2 * (values[aa] - dotprod) * designMatrix[aa].get(k);
                }
            }

            for (int kk = 0; kk < K; kk++) {
                grads[kk] += 2 * topL * w[kk];
            }

            for (int vv = 0; vv < V; vv++) {
                grads[K + vv] += 2 * lexL * w[K + vv];
            }
            return grads;
        }
    }
}

class LexicalSLDATestRunner implements Runnable {

    LexicalSLDAIdealPoint sampler;
    File stateFile;
    ArrayList<Integer> testDocIndices;
    int[][] testWords;
    int[] testAuthors;
    ArrayList<Integer> testAuthorIndices;
    boolean[][] testVotes;
    File predictionFile;

    public LexicalSLDATestRunner(LexicalSLDAIdealPoint sampler,
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
        LexicalSLDAIdealPoint testSampler = new LexicalSLDAIdealPoint();
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
