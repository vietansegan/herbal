package votepredictor.textidealpoint.backup;

import cc.mallet.optimize.LimitedMemoryBFGS;
import data.Author;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import optimization.RidgeLinearRegressionOptimizable;
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
public class HierSingleTIPM extends AbstractTextSingleIdealPoint {

    protected double topicAlpha;
    protected double frameAlpha;
    protected double topicBeta;
    protected double frameBeta;

    protected double gamma; // eta variance

    // input
    protected int K;
    protected int J; // number of frames
    protected double[][] priorTopics;

    // latent variables
    protected DirMult[] topicWords;
    protected DirMult[][] frameWords;
    protected DirMult[] docTopics;
    protected DirMult[][] docFrames;

    protected int[][] topicZs;
    protected int[][] frameZs;
    protected double[][] etas; // regression parameters for topics
    protected ArrayList<double[][]> etaList;
    protected double[] authorMeans;

    protected SparseCount[][] authorFrameCounts;  // per author
    protected SparseCount[] authorTopicCounts;    // per author

    private int numFrameAssignmentChange;
    private ArrayList<String> topicVocab;
    private double sqrtRho;

    public HierSingleTIPM() {
        this.basename = "Hier-Single-TIPM";
    }

    public HierSingleTIPM(String bname) {
        this.basename = bname;
    }

    public void setTopicVocab(ArrayList<String> topicVoc) {
        this.topicVocab = topicVoc;
    }

    public int getIndex(int kk, int jj) {
        return kk * J + jj;
    }

    public double[] getPredictedUs() {
        return this.authorMeans;
    }

    public double[][] getMultipleUs() {
        SparseCount[][] frameCounts = new SparseCount[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                frameCounts[aa][kk] = new SparseCount();
            }
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                frameCounts[aa][topicZs[dd][nn]].increment(frameZs[dd][nn]);
            }
        }

        double[][] us = new double[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                us[aa][kk] = frameCounts[aa][kk].dotprod(etas[kk]) / frameCounts[aa][kk].getCountSum();
            }
        }
        return us;
    }

    public void configure(
            String folder,
            int V, int K, int J,
            double topicAlpha, double frameAlpha,
            double topicBeta, double frameBeta,
            double rho,
            double sigma,
            double gamma,
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
        this.etaList = new ArrayList<>();

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
                .append("_g-").append(formatter.format(this.gamma));
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
            logln("--- burn-in:\t" + BURN_IN);
            logln("--- max iter:\t" + MAX_ITER);
            logln("--- sample lag:\t" + LAG);
            logln("--- paramopt:\t" + paramOptimized);
            logln("--- initialize:\t" + initState);
            logln("--- has prior? " + (priorTopics != null));
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

        initializeIdealPoint();
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

        authorMeans = new double[A];
        for (int aa = 0; aa < A; aa++) {
            if (authorTotalWordWeights[aa] == 0.0 && u != null) {
                authorMeans[aa] = u[aa];
            }
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

        // initialize by sampling without using the metada
        for (int ii = 0; ii < 50; ii++) {
            sampleZs(REMOVE, ADD, REMOVE, ADD, !OBSERVED);
        }

        outputTopicTopWords(new File(getSamplerFolderPath(), TopWordFile + ".init"), 15);
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
                    double dotprod = authorMeans[aa] * x[bb] + y[bb];
                    double score = Math.exp(dotprod);
                    double prob = score / (1.0 + score);
                    predictions[author].set(bill, prob);
                }
            }
        }
        return predictions;
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
        double absU = 0.0;
        double absMean = 0.0;
        for (int aa = 0; aa < A; aa++) {
            double diff = u[aa] - authorMeans[aa];
            mse += diff * diff;
            absU += Math.abs(u[aa]);
            absMean += Math.abs(authorMeans[aa]);
        }
        logln("--- Regression MSE: " + mse / A);
        logln("--- Abs U: " + absU / A);
        logln("--- Abs mean U: " + absMean / A);
        return str;
    }

    @Override
    public void iterate() {
        updateEtas();

        sampleZs(REMOVE, ADD, REMOVE, ADD, OBSERVED);

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
                    authorMeans[aa] -= etas[topicZs[d][n]][frameZs[d][n]] / authorTotalWordWeights[aa];
                }

                // sample topic
                double[] probs = new double[K];
                for (int kk = 0; kk < K; kk++) {
                    probs[kk] = docTopics[d].getProbability(kk)
                            * topicWords[kk].getProbability(words[d][n]);
                }
                int st = SamplerUtils.scaleSample(probs); // sampled topic

                // sample frame
                double[] logprobs = new double[J];
                for (int jj = 0; jj < J; jj++) {
                    logprobs[jj]
                            = Math.log(docFrames[d][st].getProbability(jj))
                            + Math.log(frameWords[st][jj].getProbability(words[d][n]));
                    if (observe) {
                        double mean = authorMeans[aa] + etas[st][jj] / authorTotalWordWeights[aa];
                        double resLlh = StatUtils.logNormalProbability(u[aa], mean, sqrtRho);
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
                    authorMeans[aa] += etas[topicZs[d][n]][frameZs[d][n]] / authorTotalWordWeights[aa];
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

        // current eta's
        double[] flatEta = new double[K * J];
        for (int kk = 0; kk < K; kk++) {
            System.arraycopy(etas[kk], 0, flatEta, kk * J, J);
        }

        SparseVector[] designMatrix = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            designMatrix[aa] = new SparseVector(K * J);
        }
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                designMatrix[aa].change(getIndex(topicZs[dd][nn], frameZs[dd][nn]),
                        1.0 / authorTotalWordWeights[aa]);
            }
        }

        RidgeLinearRegressionOptimizable optimizable = new RidgeLinearRegressionOptimizable(
                u, flatEta, designMatrix, rho, 0.0, gamma);

        LimitedMemoryBFGS optimizer = new LimitedMemoryBFGS(optimizable);
        boolean converged = false;
        try {
            converged = optimizer.optimize();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        // update regression parameters
        int idx = 0;
        flatEta = new double[K * J];
        for (int kk = 0; kk < K; kk++) {
            for (int jj = 0; jj < J; jj++) {
                etas[kk][jj] = optimizable.getParameter(idx);
                flatEta[idx] = etas[kk][jj];
                idx++;
            }
        }

        for (int aa = 0; aa < A; aa++) {
            authorMeans[aa] = designMatrix[aa].dotProduct(flatEta); // for voters who don't talk
            if (authorTotalWordWeights[aa] == 0) {
                if (authorMeans[aa] != 0) {
                    throw new MismatchRuntimeException(Double.toString(authorMeans[aa]), "0.0");
                }
                authorMeans[aa] = u[aa];
            }
        }

        if (iter > BURN_IN && iter % LAG == 0) {
            this.etaList.add(cloneEtas());
        }

        long eTime = System.currentTimeMillis() - sTime;
        if (isReporting) {
            logln("--- --- converged? " + converged);
            logln("--- --- time: " + eTime);
        }
        return eTime;
    }

    private double[][] cloneEtas() {
        double[][] cloneEtas = new double[K][J];
        for (int kk = 0; kk < K; kk++) {
            cloneEtas[kk] = etas[kk].clone();
        }
        return cloneEtas;
    }

    /**
     * Update ideal point model's parameters using gradient ascent.
     *
     * @return Elapsed time
     */
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
            grad -= (u[aa] - authorMeans[aa]) / rho; // prior
            u[aa] += aRate * grad; // update
        }
    }

    @Override
    public void updateXYs() {
        double bRate = getLearningRate();
        for (int bb = 0; bb < B; bb++) {
            double gradX = 0.0;
            double gradY = 0.0;
            // likelihood
            for (int aa = 0; aa < A; aa++) {
                if (isValidVote(aa, bb)) {
                    double score = Math.exp(u[aa] * x[bb] + y[bb]);
                    gradX += u[aa] * (getVote(aa, bb) - score / (1 + score));
                    gradY += getVote(aa, bb) - score / (1 + score);
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
                authorStr.append(u[aa]).append("\n");
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
                }
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing assignments from "
                    + zipFilepath);
        }
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
                    if (this.etaList != null) {
                        writer.write("\n");
                        for (double[][] e : this.etaList) {
                            writer.write("\t" + e[kk][jj]);
                        }
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

    public void outputTheta(File outputFile) {
        if (verbose) {
            logln("Output thetas to " + outputFile);
        }
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Topic\tCount\n");
            for (int kk = 0; kk < K; kk++) {
                String topicLabel = "Topic-" + kk;
                if (this.topicVocab != null) {
                    topicLabel = "\"" + this.topicVocab.get(kk) + "\"";
                }
                writer.write(topicLabel + "\t" + topicWords[kk].getCountSum() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
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

        SparseCount[][] authorFrameCounts = new SparseCount[A][K];
        for (int aa = 0; aa < A; aa++) {
            for (int kk = 0; kk < K; kk++) {
                authorFrameCounts[aa][kk] = new SparseCount();
            }
        }

        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int kk = 0; kk < K; kk++) {
                for (int jj : docFrames[dd][kk].getSparseCounts().getIndices()) {
                    int count = docFrames[dd][kk].getCount(jj);
                    authorFrameCounts[aa][kk].changeCount(jj, count);
                }
            }
        }

        // validate
        for (int aa = 0; aa < A; aa++) {
            int authorTokens = 0;
            for (int kk = 0; kk < K; kk++) {
                authorTokens += authorFrameCounts[aa][kk].getCountSum();
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
                            val = (double) authorFrameCounts[aa][kk].getCount(jj) / authorTotalWordWeights[aa];
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
                                + "\t" + u[aa]
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

    private String getTopicLabel(int kk) {
        return "\"" + this.topicVocab.get(kk) + "\"";
    }
}

//    protected long updateEtasOWLQN() {
//        if (isReporting) {
//            logln("+++ Updating etas ...");
//        }
//        long sTime = System.currentTimeMillis();
//
//        // current eta's
//        double[] flatEta = new double[K * J];
//        for (int kk = 0; kk < K; kk++) {
//            System.arraycopy(etas[kk], 0, flatEta, kk * J, J);
//        }
//
//        SparseVector[] designMatrix = new SparseVector[A];
//        for (int aa = 0; aa < A; aa++) {
//            designMatrix[aa] = new SparseVector(K * J);
//        }
//        for (int dd = 0; dd < D; dd++) {
//            int aa = authors[dd];
//            for (int nn = 0; nn < words[dd].length; nn++) {
//                int kk = topicZs[dd][nn];
//                int jj = frameZs[dd][nn];
//                designMatrix[aa].change(getIndex(kk, jj), 1.0 / authorTotalWordWeights[aa]);
//            }
//        }
//
//        // optimize
//        OWLQNLinearRegression opt = new OWLQNLinearRegression(basename, 0.1, 0.1);
//        opt.setQuiet(true);
//        OWLQNLinearRegression.setVerbose(false);
//        opt.train(designMatrix, u, flatEta);
//
//        for (int kk = 0; kk < K; kk++) {
//            for (int jj = 0; jj < J; jj++) {
//                etas[kk][jj] = flatEta[getIndex(kk, jj)];
//            }
//        }
//
//        for (int aa = 0; aa < A; aa++) {
//            authorMeans[aa] = designMatrix[aa].dotProduct(flatEta); // for voters who don't talk
//            if (authorTotalWordWeights[aa] == 0) {
//                if (authorMeans[aa] != 0) {
//                    throw new MismatchRuntimeException(Double.toString(authorMeans[aa]), "0.0");
//                }
//                authorMeans[aa] = u[aa];
//            }
//        }
//
//        long eTime = System.currentTimeMillis() - sTime;
//        if (isReporting) {
//            logln("--- --- time: " + eTime);
//        }
//        return eTime;
//    }
