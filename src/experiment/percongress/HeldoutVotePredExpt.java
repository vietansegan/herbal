package experiment.percongress;

import data.Congress;
import data.TextDataset;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;
import org.apache.commons.cli.ParseException;
import util.CLIUtils;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class HeldoutVotePredExpt extends VotePredExpt {

    @Override
    public String getConfiguredExptFolder() {
        return "heldout-vote-cv-" + numFolds + "-" + teRatio + "-" + trToDevRatio;
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

            inputCrossValidatedVotes(ff);

            runModel(foldFolder);
        }
        evaluate();
    }

    @Override
    public void preprocess() {
        if (verbose) {
            logln("Preprocessing ...");
        }
        try {
            loadFormattedData();
            outputCrossValidatedVotes();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while preprocessing");
        }
    }

    /**
     * Create and output cross-validated data.
     *
     * @throws java.lang.Exception
     */
    private void outputCrossValidatedVotes() throws Exception {
        File cvFolder = new File(processedDataFolder, getConfiguredExptFolder());
        IOUtils.createFolder(cvFolder);

        if (verbose) {
            logln("--- Outputing cross-validated data to " + cvFolder);
        }

        Random rand = new Random(1);
        double[] probs = new double[2];
        probs[0] = teRatio; // for test
        probs[1] = probs[0] + (1.0 - probs[0]) * trToDevRatio;
        for (int ff = 0; ff < numFolds; ff++) {
            BufferedWriter writer = IOUtils.getBufferedWriter(new File(cvFolder,
                    "fold-" + ff + ".dat"));
            writer.write(votes.length + "\n");

            int numTrain = 0;
            int numDev = 0;
            int numTest = 0;

            trainVotes = new boolean[votes.length][];
            devVotes = new boolean[votes.length][];
            testVotes = new boolean[votes.length][];

            for (int aa = 0; aa < votes.length; aa++) {
                writer.write(aa + "\t" + votes[aa].length);
                for (int vv = 0; vv < votes[aa].length; vv++) {
                    if (votes[aa][vv] == Vote.MISSING) {
                        writer.write("\t" + Vote.MISSING);
                        continue;
                    }
                    double val = rand.nextDouble();
                    if (val < probs[0]) {
                        writer.write("\t" + TEST_POSFIX);
                        numTest++;
                    } else if (val < probs[1]) {
                        writer.write("\t" + TRAIN_POSFIX);
                        numTrain++;
                    } else {
                        writer.write("\t" + DEV_POSFIX);
                        numDev++;
                    }
                }
                writer.write("\n");
            }

            if (verbose) {
                logln("--- Fold " + ff);
                logln("--- --- # training votes: " + numTrain);
                logln("--- --- # development votes: " + numDev);
                logln("--- --- # test votes: " + numTest);
            }

            writer.close();
        }
    }

    /**
     * Load cross-validated data.
     *
     * @param ff Fold index
     */
    private void inputCrossValidatedVotes(int ff) {
        File cvFolder = new File(processedDataFolder, getConfiguredExptFolder());
        try {
            BufferedReader reader = IOUtils.getBufferedReader(new File(cvFolder,
                    "fold-" + ff + ".dat"));
            int numAuthors = Integer.parseInt(reader.readLine());

            trainVotes = new boolean[numAuthors][];
            devVotes = new boolean[numAuthors][];
            testVotes = new boolean[numAuthors][];
            int numTrain = 0;
            int numDev = 0;
            int numTest = 0;
            for (int aa = 0; aa < numAuthors; aa++) {
                String[] sline = reader.readLine().split("\t");
                if (aa != Integer.parseInt(sline[0])) {
                    throw new RuntimeException("Mismatch");
                }
                int voteLength = Integer.parseInt(sline[1]);
                if (voteLength != sline.length - 2) {
                    throw new RuntimeException("Mismatch. " + voteLength
                            + " vs. " + (sline.length - 2));
                }
                trainVotes[aa] = new boolean[voteLength];
                devVotes[aa] = new boolean[voteLength];
                testVotes[aa] = new boolean[voteLength];
                for (int ii = 0; ii < voteLength; ii++) {
                    switch (sline[ii + 2]) {
                        case TRAIN_POSFIX:
                            trainVotes[aa][ii] = true;
                            numTrain++;
                            break;
                        case DEV_POSFIX:
                            devVotes[aa][ii] = true;
                            numDev++;
                            break;
                        case TEST_POSFIX:
                            testVotes[aa][ii] = true;
                            numTest++;
                            break;
                    }
                }
            }
            reader.close();

            this.trainAuthorIndices = new ArrayList<>();
            for (int aa = 0; aa < numAuthors; aa++) {
                this.trainAuthorIndices.add(aa);
            }

            this.trainBillIndices = new ArrayList<>();
            for (int bb = 0; bb < trainVotes[0].length; bb++) {
                this.trainBillIndices.add(bb);
            }

            if (verbose) {
                logln("--- Fold " + ff + " loaded");
                logln("--- --- # training votes: " + numTrain);
                logln("--- --- # development votes: " + numDev);
                logln("--- --- # test votes: " + numTest);
            }
        } catch (IOException | RuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing cross-validated votes");
        }
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOptions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(HeldoutVotePredExpt.class
                        .getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            HeldoutVotePredExpt expt = new HeldoutVotePredExpt();
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
