package experiment.percongress;

import data.Congress;
import data.TextDataset;
import data.Vote;
import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import org.apache.commons.cli.ParseException;
import util.CLIUtils;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class GOPExpt extends VotePredExpt {

    @Override
    public String getConfiguredExptFolder() {
        return "gop";
    }

    /**
     * Estimate on all data.
     */
    @Override
    public void run() {
        if (verbose) {
            logln("Running models ...");
        }

        setupSampling();

        loadFormattedData();

        trainAuthorIndices = new ArrayList<>();
        for (int aa = 0; aa < debateVoteData.getAuthorVocab().size(); aa++) {
            String authorId = debateVoteData.getAuthorVocab().get(aa);
            String party = debateVoteData.getAuthorProperty(authorId, "party");
            if (party.equals("Republican")) {
                trainAuthorIndices.add(aa);
            }
        }

        trainVotes = new boolean[votes.length][];
        for (int aa = 0; aa < votes.length; aa++) {
            trainVotes[aa] = new boolean[votes[aa].length];
            Arrays.fill(trainVotes[aa], false);
            if (trainAuthorIndices.contains(aa)) {
                for (int bb = 0; bb < votes[aa].length; bb++) {
                    if (votes[aa][bb] != Vote.MISSING) {
                        trainVotes[aa][bb] = true;
                    }
                }
            }
        }

        trainDebateIndices = new ArrayList<>();
        for (int dd = 0; dd < debateVoteData.getWords().length; dd++) {
            int author = debateVoteData.getAuthors()[dd];
            if (trainAuthorIndices.contains(author)) {
                this.trainDebateIndices.add(dd);
            }
        }

        if (verbose) {
            logln("--- # Republicans: " + trainAuthorIndices.size()
                    + " / " + debateVoteData.getAuthorVocab().size());
            logln("--- # debates: " + this.trainDebateIndices.size()
                    + " / " + debateVoteData.getWords().length);
        }

        File estimateFolder = new File(new File(experimentPath, congressNum),
                getConfiguredExptFolder());
        IOUtils.createFolder(estimateFolder);
        this.runModel(estimateFolder);
    }

    public static void main(String[] args) {
        try {
            long sTime = System.currentTimeMillis();

            addOptions();

            cmd = parser.parse(options, args);
            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(
                        GOPExpt.class.getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            GOPExpt expt = new GOPExpt();
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            switch (runMode) {
                case "run":
                    expt.run();
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
