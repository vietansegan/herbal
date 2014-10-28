package experiment.percongress;

import static core.AbstractExperiment.MODEL_FILE;
import static core.AbstractExperiment.RESULT_FILE;
import static core.AbstractExperiment.RESULT_FOLDER;
import static core.AbstractExperiment.TEST_PREFIX;
import core.AbstractModel;
import static core.AbstractRunner.logln;
import data.Congress;
import data.TextDataset;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Random;
import org.apache.commons.cli.ParseException;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.IOUtils;
import util.SparseVector;
import votepredictor.AbstractVotePredictor;
import votepredictor.baselines.AuthorTFIDFNN;
import votepredictor.baselines.AuthorTFNN;
import votepredictor.baselines.LogisticRegression;

/**
 *
 * @author vietan
 */
public class HeldoutAuthorPredExpt extends VotePredExpt {

    @Override
    public String getConfiguredExptFolder() {
        return "author-heldout-" + numFolds + "-" + teRatio;
    }

    @Override
    public void run() {
        if (verbose) {
            logln("Running ...");
        }
        ArrayList<Integer> runningFolds = new ArrayList<Integer>();
        if (cmd.hasOption("fold")) {
            String foldList = cmd.getOptionValue("fold");
            for (String f : foldList.split(",")) {
                runningFolds.add(Integer.parseInt(f));
            }
        }

        loadFormattedData();

        setupSampling();

        File configureFolder = new File(new File(experimentPath, congressNum),
                getConfiguredExptFolder());

        for (int ff = 0; ff < numFolds; ff++) {
            if (!runningFolds.isEmpty() && !runningFolds.contains(ff)) {
                continue;
            }
            if (verbose) {
                logln("--- Running fold " + ff);
            }

            File foldFolder = new File(configureFolder, "fold-" + ff);
            IOUtils.createFolder(foldFolder);

            inputCrossValidatedData(ff);

            runModel(foldFolder);
        }
    }

    /**
     * Run a model.
     *
     * @param outputFolder Output folder
     */
    @Override
    protected void runModel(File outputFolder) {
        String model = CLIUtils.getStringArgument(cmd, "model", "random");
        switch (model) {
            case "random":
                runRandom(outputFolder);
                break;
            case "log-reg":
                runLogisticRegressors(outputFolder);
                break;
            case "author-tf-nn":
                runAuthorTFNN(outputFolder);
                break;
            case "author-tf-idf-nn":
                runAuthorTFIDFNN(outputFolder);
                break;
            case "slda-ideal-point":
                runSLDAIdealPoint(outputFolder);
                break;
            case "snlda-ideal-point":
                runSNLDAIdealPoint(outputFolder);
                break;
            case "snhdp-ideal-point":
                runSNHDPIdealPoint(outputFolder);
                break;
            case "none":
                logln("Doing nothing :D");
                break;
            default:
                throw new RuntimeException("Model " + model + " not supported");
        }
    }

    protected void runLogisticRegressors(File outputFolder) {
        if (verbose) {
            logln("--- --- Running logistic regressors ...");
        }
        double mu = CLIUtils.getDoubleArgument(cmd, "mu", 0.0);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 1.0);

        LogisticRegression lr = new LogisticRegression("logreg-lbfgs");
        lr.configure(debateVoteData.getWordVocab().size(), mu, sigma);
        File predFolder = new File(outputFolder, lr.getName());
        IOUtils.createFolder(predFolder);

        if (cmd.hasOption("train")) {
            lr.train(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            lr.output(new File(predFolder, MODEL_FILE));
        }

        if (cmd.hasOption("testauthor")) {
            lr.input(new File(predFolder, MODEL_FILE));

            SparseVector[] predictions = lr.test(
                    testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    testAuthorIndices,
                    testVotes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }
    }

    protected void runAuthorTFNN(File outputFolder) {
        if (verbose) {
            logln("--- --- Running author TF-NN ...");
        }
        int K = CLIUtils.getIntegerArgument(cmd, "K", 5);
        AuthorTFNN pred = new AuthorTFNN("author-tf");
        pred.configure(debateVoteData.getWordVocab().size());
        File predFolder = new File(outputFolder, pred.getName());
        IOUtils.createFolder(predFolder);

        if (cmd.hasOption("train")) {
            pred.train(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            pred.output(new File(predFolder, MODEL_FILE));
        }

        if (cmd.hasOption("testauthor")) {
            pred.input(new File(predFolder, MODEL_FILE));

            SparseVector[] predictions = pred.test(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    testAuthorIndices,
                    testVotes, K,
                    votes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }
    }

    protected void runAuthorTFIDFNN(File outputFolder) {
        if (verbose) {
            logln("--- --- Running author TF-IDF-NN ...");
        }
        int K = CLIUtils.getIntegerArgument(cmd, "K", 5);
        AuthorTFIDFNN pred = new AuthorTFIDFNN("author-tf-idf");
        pred.configure(debateVoteData.getWordVocab().size());
        File predFolder = new File(outputFolder, pred.getName());
        IOUtils.createFolder(predFolder);

        if (cmd.hasOption("train")) {
            pred.train(trainDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    votes,
                    trainAuthorIndices,
                    trainBillIndices,
                    trainVotes);
            pred.output(new File(predFolder, MODEL_FILE));
        }

        if (cmd.hasOption("testauthor")) {
            pred.input(new File(predFolder, MODEL_FILE));

            SparseVector[] predictions = pred.test(testDebateIndices,
                    debateVoteData.getWords(),
                    debateVoteData.getAuthors(),
                    testAuthorIndices,
                    testVotes, K,
                    votes);
            File teResultFolder = new File(predFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            AbstractModel.outputPerformances(new File(teResultFolder, RESULT_FILE),
                    AbstractVotePredictor.evaluate(votes, testVotes, predictions));
        }
    }

    @Override
    public void preprocess() {
        if (verbose) {
            logln("Preprocessing ...");
        }
        try {
            loadFormattedData();
            outputCrossValidatedData();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while preprocessing");
        }
    }

    private void outputCrossValidatedData() throws Exception {
        File cvFolder = new File(processedDataFolder, getConfiguredExptFolder());
        IOUtils.createFolder(cvFolder);

        if (verbose) {
            logln("--- Outputing cross-validated data to " + cvFolder);
        }

        Random rand = new Random(1);

        int A = debateVoteData.getAuthorVocab().size();
        int D = debateVoteData.getWords().length;

        // list of author indices
        ArrayList<Integer> authorIndices = new ArrayList<>();
        for (int aa = 0; aa < A; aa++) {
            authorIndices.add(aa);
        }

        // count number of documents per author
        SparseCount authorDocCount = new SparseCount();
        for (int dd = 0; dd < D; dd++) {
            authorDocCount.increment(this.debateVoteData.getAuthors()[dd]);
        }

        if (verbose) {
            logln("--- --- # authors: " + A);
            logln("--- --- # authors with text: " + authorDocCount.size());
        }

        BufferedWriter writer;
        for (int ff = 0; ff < numFolds; ff++) {
            this.trainAuthorIndices = new ArrayList<>();
            this.testAuthorIndices = new ArrayList<>();
            Collections.shuffle(authorIndices);

            for (int ii = 0; ii < A; ii++) {
                int aa = authorIndices.get(ii);
                if (!authorDocCount.containsIndex(aa)) {
                    this.trainAuthorIndices.add(aa);
                } else {
                    if (rand.nextDouble() < teRatio) {
                        this.testAuthorIndices.add(aa);
                    } else {
                        this.trainAuthorIndices.add(aa);
                    }
                }
            }

            if (verbose) {
                logln("--- --- Fold " + ff);
                logln("--- --- # train authors: " + trainAuthorIndices.size());
                logln("--- --- # test authors: " + testAuthorIndices.size());
            }

            writer = IOUtils.getBufferedWriter(new File(cvFolder, "fold-" + ff + ".dat"));
            for (int aa = 0; aa < A; aa++) {
                if (trainAuthorIndices.contains(aa)) {
                    writer.write(aa + "\t" + TRAIN_POSFIX + "\n");
                } else {
                    writer.write(aa + "\t" + TEST_POSFIX + "\n");
                }
            }
            writer.close();
        }
    }

    protected void inputCrossValidatedData(int ff) {
        int A = debateVoteData.getAuthorVocab().size();
        int B = debateVoteData.getVoteVocab().size();
        int D = debateVoteData.getWords().length;

        this.trainAuthorIndices = new ArrayList<>();
        this.trainDebateIndices = new ArrayList<>();
        this.trainVotes = new boolean[A][B];
        this.trainBillIndices = null; // use all bills

        this.testAuthorIndices = new ArrayList<>();
        this.testDebateIndices = new ArrayList<>();
        this.testVotes = new boolean[A][B];
        this.testBillIndices = null; // use all bills

        File cvFolder = new File(processedDataFolder, getConfiguredExptFolder());
        try {
            if (verbose) {
                logln("--- Loading fold " + ff);
            }

            BufferedReader reader = IOUtils.getBufferedReader(new File(cvFolder,
                    "fold-" + ff + ".dat"));
            for (int aa = 0; aa < A; aa++) {
                String[] sline = reader.readLine().split("\t");
                if (aa != Integer.parseInt(sline[0])) {
                    throw new RuntimeException("Mismatch");
                }
                if (sline[1].equals(TRAIN_POSFIX)) {
                    this.trainAuthorIndices.add(aa);
                } else {
                    this.testAuthorIndices.add(aa);
                }
            }
            reader.close();

            for (int dd = 0; dd < D; dd++) {
                int aa = debateVoteData.getAuthors()[dd];
                if (trainAuthorIndices.contains(aa)) {
                    trainDebateIndices.add(dd);
                } else {
                    testDebateIndices.add(dd);
                }
            }

            int numTrainVotes = 0;
            int numTestVotes = 0;
            for (int ii = 0; ii < A; ii++) {
                for (int jj = 0; jj < B; jj++) {
                    if (debateVoteData.getVotes()[ii][jj] == Vote.MISSING) {
                        this.trainVotes[ii][jj] = false;
                        this.testVotes[ii][jj] = false;
                    } else if (trainAuthorIndices.contains(ii)) {
                        this.trainVotes[ii][jj] = true;
                        numTrainVotes++;
                    } else {
                        this.testVotes[ii][jj] = true;
                        numTestVotes++;
                    }
                }
            }

            if (verbose) {
                logln("--- --- train. # authors: " + trainAuthorIndices.size()
                        + ". # documents: " + trainDebateIndices.size()
                        + ". # votes: " + numTrainVotes);
                logln("--- --- test. # authors: " + testAuthorIndices.size()
                        + ". # documents: " + testDebateIndices.size()
                        + ". # votes: " + numTestVotes);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing fold " + ff);
        }
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOptions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(HeldoutAuthorPredExpt.class
                        .getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            HeldoutAuthorPredExpt expt = new HeldoutAuthorPredExpt();
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            switch (runMode) {
                case "run":
                    expt.run();
                    break;
                case "evaluate":
                    expt.evaluate();
                    break;
                case "create-cv":
                    expt.preprocess();
                    break;
                default:
                    throw new RuntimeException("Run mode " + runMode + " is not supported");
            }

            // date and time
            DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
            Date dateobj = new Date();
            long eTime = (System.currentTimeMillis() - sTime) / 1000;
            System.out.println("Elapsed time: " + eTime + "s");
            System.out.println("End time: " + df.format(dateobj));
        } catch (RuntimeException | ParseException e) {
            e.printStackTrace();
        }
    }
}
