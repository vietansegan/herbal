package experiment.percongress;

import data.Congress;
import data.TextDataset;
import data.Vote;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import org.apache.commons.cli.ParseException;
import util.CLIUtils;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class HeldoutAuthorDebatePredExpt extends HeldoutAuthorPredExpt {

    @Override
    public String getConfiguredExptFolder() {
        return "author-heldout-debates-" + numFolds + "-" + teRatio;
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

        getKeyvoteBills();

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

        this.evaluate();
    }

    @Override
    public void analyze() {
        if (verbose) {
            logln("Analyzing ...");
        }
        ArrayList<Integer> runningFolds = new ArrayList<Integer>();
        if (cmd.hasOption("fold")) {
            String foldList = cmd.getOptionValue("fold");
            for (String f : foldList.split(",")) {
                runningFolds.add(Integer.parseInt(f));
            }
        }

        loadFormattedData();

        getKeyvoteBills();

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

            analyzeErrorMultipleModels(foldFolder);
        }
    }

    /**
     * Input the cross-validated data from a fold.
     *
     * @param ff Fold number
     */
    @Override
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

        File cvFolder = new File(processedDataFolder, super.getConfiguredExptFolder());
        try {
            if (verbose) {
                logln("--- Loading fold " + ff + " from " + cvFolder);
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
                String billId = debateVoteData.getBillIds()[dd];
                if (trainAuthorIndices.contains(aa)) {
                    if (!this.keyvoteBills.contains(billId)) {
                        trainDebateIndices.add(dd);
                    }
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
                logln("--- --- # key votes: " + this.keyvoteBills.size());
                logln("--- --- train. # authors: " + trainAuthorIndices.size()
                        + ". # documents: " + trainDebateIndices.size()
                        + ". # votes: " + numTrainVotes);
                logln("--- --- test. # authors: " + testAuthorIndices.size()
                        + ". # documents: " + testDebateIndices.size()
                        + ". # votes: " + numTestVotes);
            }
        } catch (IOException | RuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing fold " + ff
                    + " from " + cvFolder);
        }
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOptions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(HeldoutAuthorDebatePredExpt.class
                        .getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            HeldoutAuthorDebatePredExpt expt = new HeldoutAuthorDebatePredExpt();
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            switch (runMode) {
                case "run":
                    expt.run();
                    break;
                case "evaluate":
                    expt.evaluate();
                    break;
                case "analyze":
                    expt.analyze();
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
