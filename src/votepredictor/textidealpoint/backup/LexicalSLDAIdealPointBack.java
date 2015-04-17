package votepredictor.textidealpoint.backup;

import cc.mallet.optimize.LimitedMemoryBFGS;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
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
import votepredictor.AbstractVotePredictor;
import votepredictor.BayesianIdealPoint;
import votepredictor.textidealpoint.AbstractTextSingleIdealPoint;

/**
 *
 * @author vietan
 */
public class LexicalSLDAIdealPointBack extends AbstractTextSingleIdealPoint {

    public static final int numSteps = 10;
    public static final int ALPHA = 0;
    public static final int BETA = 1;
    public double rho;
    public double sigma;
    public double gamma; // eta variance
    public double lambda; // l2-norm using L-BFGS
    public double l1; // l1-norm using OWL-QN
    public double l2; // l2-norm using OWL-QN
    // input
    protected int K; // number of topics
    // derived
    protected SparseVector[] wa;
    protected double[] idfs;
    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[] docTopics;
    protected int[][] z;
    protected double[] eta; // regression parameters for topics
    protected double[] tau; // lexical regression parameters
    // internal
    protected double[] zaEta;
    protected double[] waTau;

    public LexicalSLDAIdealPointBack() {
        this.basename = "Lexical-SLDA-ideal-point";
    }

    public LexicalSLDAIdealPointBack(String bname) {
        this.basename = bname;
    }

    public void configure(LexicalSLDAIdealPointBack sampler) {
        this.configure(sampler.folder,
                sampler.V,
                sampler.K,
                sampler.hyperparams.get(ALPHA),
                sampler.hyperparams.get(BETA),
                sampler.rho,
                sampler.sigma,
                sampler.lambda,
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
            double lambda,
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
        this.lambda = lambda;
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
        this.setName();

        this.report = true;

        if (verbose) {
            logln("--- folder\t" + folder);
            logln("--- num topics:\t" + K);
            logln("--- num word types:\t" + V);
            logln("--- alpha:\t" + MiscUtils.formatDouble(hyperparams.get(ALPHA)));
            logln("--- beta:\t" + MiscUtils.formatDouble(hyperparams.get(BETA)));
            logln("--- rho:\t" + MiscUtils.formatDouble(rho));
            logln("--- reg sigma:\t" + MiscUtils.formatDouble(sigma));
            logln("--- lambda:\t" + MiscUtils.formatDouble(lambda));
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
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
                .append("_l-").append(formatter.format(lambda))
                .append("_r-").append(formatter.format(rho))
                .append("_s-").append(formatter.format(sigma));
        str.append("_opt-").append(this.paramOptimized);
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

        if (this.idfs == null) {
            this.idfs = MiscUtils.getIDFs(words, V);
        }

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
                this.wa[aa].change(vv, Math.log(count + 1) * idfs[vv]);
            }
        }

        for (int aa = 0; aa < A; aa++) {
            this.wa[aa].normalize();
        }
    }

    @Override
    public String getCurrentState() {
        StringBuilder str = new StringBuilder();
        str.append(this.getSamplerFolderPath())
                .append("\nCurrent thread: ").append(Thread.currentThread().getId());
        return str.toString();
    }

    @Override
    public void initialize() {
        initialize(null);
    }

    public void initialize(double[][] seededTopics) {
        if (verbose) {
            logln("Initializing ...");
        }
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

        BayesianIdealPoint bip = new BayesianIdealPoint();
        bip.configure(1.0, 0.01, 5000, 0.0, sigma);
        bip.setTrain(votes, authorIndices, billIndices, validVotes);

        File bipFolder = new File(folder, bip.getName());
        File bipFile = new File(bipFolder, ModelFile);

        if (bipFile.exists()) {
            if (verbose) {
                logln("B.I.P. file exists. Loading from " + bipFile);
            }
            bip.input(bipFile);
        } else {
            if (verbose) {
                logln("B.I.P. file not found. Running and outputing to " + bipFile);
            }
            IOUtils.createFolder(bipFolder);
            bip.train();
            bip.output(bipFile);
        }
        this.u = bip.getUs();
        this.x = bip.getXs();
        this.y = bip.getYs();

        this.eta = new double[K];
        this.tau = new double[V];
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
        zaEta = new double[A];
        waTau = new double[A];
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

        // run LDA
        int lda_burnin = 250;
        int lda_maxiter = 500;
        int lda_samplelag = 25;
        LDA lda = new LDA();
        lda.setDebug(false);
        lda.setVerbose(verbose);
        lda.setLog(false);
        double lda_alpha = hyperparams.get(ALPHA);
        double lda_beta = hyperparams.get(BETA);

        lda.configure(folder, V, K, lda_alpha, lda_beta, InitialState.RANDOM, false,
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
                zaEta[aa] += eta[z[dd][nn]] / authorTotalWordWeights[aa];
            }
        }
    }

    /**
     * Make prediction on held-out votes of known legislators and known bills,
     * averaging over multiple models.
     *
     * @return Predictions
     */
    public SparseVector[] predictInMatrixMultiples() {
        SparseVector[] predictions = null;
        int count = 0;
        File reportFolder = new File(this.getReportFolderPath());
        String[] files = reportFolder.list();
        for (String file : files) {
            if (!file.endsWith(".zip")) {
                continue;
            }
            this.inputState(new File(reportFolder, file));
            SparseVector[] partPreds = this.predictInMatrix();
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

    protected double getMSE() {
        double mse = 0.0;
        for (int aa = 0; aa < A; aa++) {
            double diff = u[aa] - (zaEta[aa] + waTau[aa]);
            mse += diff * diff;
        }
        return mse / A;
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
        inputBillScore(stateFile.getAbsolutePath());
        initializeDataStructure();

        waTau = new double[A];
        for (int aa = 0; aa < A; aa++) {
            waTau[aa] = wa[aa].dotProduct(tau);
        }

        // sample
        ArrayList<SparseVector[]> predictionList = new ArrayList<>();
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
                logln("--- Evaluating ...");
                SparseVector[] predictions = predictInMatrix();
                ArrayList<Measurement> measurements = AbstractVotePredictor
                        .evaluateAll(votes, validVotes, predictions);
                for (Measurement m : measurements) {
                    logln(">>> i >>> " + m.getName() + ": " + m.getValue());
                }

                predictions = predictOutMatrix();
                measurements = AbstractVotePredictor
                        .evaluateAll(votes, validVotes, predictions);
                for (Measurement m : measurements) {
                    logln(">>> o >>> " + m.getName() + ": " + m.getValue());
                }
                logln("--- MSE: " + getMSE());
            }

            // update lexical regression parameters
            if (lambda > 0 && iter % LAG == 0) {
                updateTaus();
            }

            // update topic regression parameters
            updateEtas();

            // sample token assignments
            sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

            // update ideal points
            if (iter >= BURN_IN && iter % LAG == 0) {
                updateUXY();
            }

            // parameter optimization
            if (iter % LAG == 0 && iter >= BURN_IN) {
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
                responses, eta, designMatrix, rho, 0.0, sigma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (isReporting) {
            logln("--- converged? " + converged);
        }

        // update regression parameters
        for (int kk = 0; kk < K; kk++) {
            eta[kk] = optimizable.getParameter(kk);
        }

        for (int aa = 0; aa < A; aa++) {
            zaEta[aa] = designMatrix[aa].dotProduct(eta);
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    public long updateTaus() {
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

        // update regression parameters
        for (int vv = 0; vv < V; vv++) {
            tau[vv] = optimizable.getParameter(vv);
        }

        // update
        for (int aa = 0; aa < A; aa++) {
            waTau[aa] = wa[aa].dotProduct(tau);
        }

        if (isReporting) {
            logln("--- converged? " + converged);
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
            grad -= (u[aa] - zaEta[aa] - waTau[aa]) / rho; // prior
            u[aa] += aRate * grad; // update
        }
    }

    private double getVoteLogLikelihood() {
        double llh = 0.0;
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    double score = u[aa] * x[bb] + y[bb];
                    llh += getVote(aa, bb) * score - Math.log(1 + Math.exp(score));
                }
            }
        }
        return llh;
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

        double voteLlh = getVoteLogLikelihood();
        double llh = wordLlh + topicLlh + voteLlh;
        if (isReporting()) {
            logln("*** word: " + MiscUtils.formatDouble(wordLlh)
                    + ". topic: " + MiscUtils.formatDouble(topicLlh)
                    + ". vote: " + MiscUtils.formatDouble(voteLlh)
                    + ". llh = " + MiscUtils.formatDouble(llh));
        }

        return llh;
    }

    @Override
    public double getLogLikelihood(ArrayList<Double> newParams) {
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

        double voteLlh = getVoteLogLikelihood();
        double llh = wordLlh + topicLlh + voteLlh;
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
            for (int kk = 0; kk < K; kk++) {
                modelStr.append(kk).append("\n");
                modelStr.append(eta[kk]).append("\n");
                modelStr.append(DirMult.output(topicWords[kk])).append("\n");
            }
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(vv).append("\t").append(this.wordWeights[vv]).append("\n");
            }
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(vv).append("\t").append(tau[vv]).append("\n");
            }
            for (int vv = 0; vv < V; vv++) {
                modelStr.append(vv).append("\t").append(idfs[vv]).append("\n");
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
            inputBillScore(filepath);
            inputAuthorScore(filepath);
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

            wordWeights = new double[V];
            for (int vv = 0; vv < V; vv++) {
                String[] sline = reader.readLine().split("\t");
                int vIdx = Integer.parseInt(sline[0]);
                if (vv != vIdx) {
                    throw new MismatchRuntimeException(vIdx, vv);
                }
                wordWeights[vv] = Double.parseDouble(sline[1]);
            }

            tau = new double[V];
            for (int vv = 0; vv < V; vv++) {
                String[] sline = reader.readLine().split("\t");
                int vIdx = Integer.parseInt(sline[0]);
                if (vv != vIdx) {
                    throw new MismatchRuntimeException(vIdx, vv);
                }
                tau[vv] = Double.parseDouble(sline[1]);
            }

            idfs = new double[V];
            for (int vv = 0; vv < V; vv++) {
                String[] sline = reader.readLine().split("\t");
                int vIdx = Integer.parseInt(sline[0]);
                if (vv != vIdx) {
                    throw new MismatchRuntimeException(vIdx, vv);
                }
                idfs[vv] = Double.parseDouble(sline[1]);
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

    public void inputBillScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading bill scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ".bill");

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

    public void inputAuthorScore(String zipFilepath) {
        if (verbose) {
            logln("--- --- Loading author scores from " + zipFilepath);
        }
        try {
            String filename = IOUtils.removeExtension(IOUtils.getFilename(zipFilepath));
            BufferedReader reader = IOUtils.getBufferedReader(zipFilepath, filename + ".author");
            int numAuthors = Integer.parseInt(reader.readLine());
            if (numAuthors != A) {
                throw new MismatchRuntimeException(numAuthors, A);
            }
            u = new double[A];
            for (int aa = 0; aa < A; aa++) {
                u[aa] = Double.parseDouble(reader.readLine());
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading model from "
                    + zipFilepath);
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
            LexicalSLDAIdealPointBack2 sampler) {
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
                LexicalSLDATestRunnerBack2 runner = new LexicalSLDATestRunnerBack2(
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