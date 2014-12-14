package experiment.percongress;

import core.AbstractExperiment;
import core.AbstractSampler.InitialState;
import data.Bill;
import data.Congress;
import data.CorpusProcessor;
import data.TextDataset;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.unsupervised.LDA;
import sampling.util.SparseCount;
import util.CLIUtils;
import util.IOUtils;
import util.RankingItem;

/**
 *
 * @author vietan
 */
public class BillTextExperiment extends AbstractExperiment<Bill> {

    protected String congressNum;
    protected String datasetFolder;
    protected String formatFolder;
    protected String modelFolder;
    protected int numTopWords;
    protected String processedDataFolder;
    public static HashMap<String, int[]> congressYearMap;

    public BillTextExperiment() {
        congressYearMap = new HashMap<String, int[]>();
        congressYearMap.put("112", new int[]{2011, 2012});
        congressYearMap.put("111", new int[]{2009, 2010});
        congressYearMap.put("110", new int[]{2007, 2008});
        congressYearMap.put("109", new int[]{2005, 2006});
    }

    @Override
    public void setup() {
        if (verbose) {
            logln("Setting up ...");
        }

        congressNum = CLIUtils.getStringArgument(cmd, "congress-num", "112");
        datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder",
                "/fs/clip-political/vietan/data/govtrack");
        numTopWords = CLIUtils.getIntegerArgument(cmd, "num-top-words", 20);
        processedDataFolder = cmd.getOptionValue("processed-data-folder");
        experimentPath = CLIUtils.getStringArgument(cmd, "expt-folder", "bill-experiments");
        modelFolder = CLIUtils.getStringArgument(cmd, "model-folder", "model");
    }

    @Override
    public void preprocess() throws Exception {
        if (verbose) {
            logln("Preprocessing ...");
        }
        if (verbose) {
            logln("--- Loading Congressional data ...");
        }
        File congressFolder = new File(new File(datasetFolder, congressNum), "processedV5");

        CorpusProcessor corpProc = TextDataset.createCorpusProcessor();
        Congress congressData = Congress.loadProcessedCongress(congressNum,
                datasetFolder,
                new File(congressFolder, "bills").getAbsolutePath(),
                null, //new File(congressFolder, "debates").getAbsolutePath(),
                new File(congressFolder, "legislators.txt").getAbsolutePath(),
                null, null, corpProc);
        data = congressData.getBillData();
        data.format(processedDataFolder);
    }

    @Override
    public void run() throws Exception {
        if (verbose) {
            logln("Run ...");
        }

        if (verbose) {
            logln("Running ...");
        }
        burn_in = CLIUtils.getIntegerArgument(cmd, "burnIn", 5);
        max_iters = CLIUtils.getIntegerArgument(cmd, "maxIter", 10);
        sample_lag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 5);
        report_interval = CLIUtils.getIntegerArgument(cmd, "report", 1);
        paramOpt = cmd.hasOption("paramOpt");
        String init = CLIUtils.getStringArgument(cmd, "init", "random");
        switch (init) {
            case "random":
                initState = InitialState.RANDOM;
                break;
            case "preset":
                initState = InitialState.PRESET;
                break;
            default:
                throw new RuntimeException("Initialization " + init + " not supported");
        }

        if (verbose) {
            logln("--- Loading data from " + processedDataFolder);
        }
        data = new Bill("bill-" + congressNum, processedDataFolder);
        data.loadFormattedData(processedDataFolder);

        File outputFolder = new File(experimentPath, congressNum);
        IOUtils.createFolder(outputFolder);
        String model = CLIUtils.getStringArgument(cmd, "model", "random");
        switch (model) {
            case "lda":
                runLDA(outputFolder);
                break;
            case "tfidf-nn":
                runTFIDFNNs(outputFolder);
                break;
            case "none":
                logln("Doing nothing :D");
                break;
            default:
                throw new RuntimeException("Model " + model + " not supported");
        }
    }

    private void runLDA(File outputFolder) {
        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);
        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        LDA sampler = new LDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(data.getWordVocab());

        sampler.configure(outputFolder.getAbsolutePath(),
                data.getWordVocab().size(), K,
                alpha, beta,
                initState, paramOpt,
                burn_in, max_iters, sample_lag, report_interval);
        File samplerFolder = new File(outputFolder, sampler.getSamplerFolder());
        IOUtils.createFolder(samplerFolder);
        sampler.train(data.getWords(), null);
        sampler.initialize();
        sampler.iterate();
        sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
    }

    private void runTFIDFNNs(File outputFolder) {
        int L = data.getTopicVocab().size();
        int V = data.getWordVocab().size();
        double[][] labelVectors = new double[L][V];
        int[] labelCounts = new int[L];
        for (int dd = 0; dd < data.getWords().length; dd++) {
            SparseCount typeCount = new SparseCount();
            for (int nn = 0; nn < data.getWords()[dd].length; nn++) {
                typeCount.increment(data.getWords()[dd][nn]);
            }

            int topic = data.getTopics()[dd];
            for (int vv : typeCount.getIndices()) {
                int count = typeCount.getCount(vv);
                labelVectors[topic][vv] += (double) count / typeCount.getCountSum();
            }
            labelCounts[topic]++;
        }

        for (int ll = 0; ll < L; ll++) {
            ArrayList<RankingItem<Integer>> rankWords = new ArrayList<>();
            for (int v = 0; v < V; v++) {
                rankWords.add(new RankingItem<Integer>(v, labelVectors[ll][v] / labelCounts[ll]));
            }
            Collections.sort(rankWords);
            System.out.println(data.getTopicVocab().get(ll));
            for (int ii = 0; ii < 20; ii++) {
                System.out.print("\t" + data.getWordVocab().get(rankWords.get(ii).getObject()));
            }
            System.out.println("\n");
        }

        for (int dd = 0; dd < 10; dd++) {
            System.out.println("dd = " + dd
                    + ". " + data.getDocIds()[dd]
                    + ". " + data.getTopics()[dd]
                    + ". " + data.getTopicVocab().get(data.getTopics()[dd]));
            for (int nn = 0; nn < 5; nn++) {
                System.out.print(data.getWordVocab().get(data.getWords()[dd][nn]) + " ");
            }
            System.out.println("\n");
        }
    }

    @Override
    public void evaluate() {
    }

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("congress-num", "Congress number");
            addOption("data-folder", "Folder storing processed data");
            addOption("format-folder", "Format folder");
            addOption("model-folder", "Model folder");
            addOption("expt-folder", "Experiment folder");
            addOption("processed-data-folder", "Processed data folder");

            addOption("tr2dev-ratio", "Training-to-developmeng ratio");

            addOption("debate-file", "Debate formatted file name");
            addOption("bill-file", "Bill formatted file name");
            addOption("legislator-file", "Legislator file");

            // files with Tea Party annotation
            addOption("house-rep-file", "House Republcian file");
            addOption("senate-rep-file", "Senate Republcian file");

            addOption("run-mode", "Run mode");
            addOption("model", "Model");

            addOption("K", "Number of topics");

            // mode parameters
            addGreekParametersOptions();

            // processing options
            addCorpusProcessorOptions();

            // sampling
            addSamplingOptions();

            // cross validation
            addCrossValidationOptions();

            // liblinear
            addOption("C", "Cost of constraints violation in LibLinear");
            addOption("solver-type", "Type of the solver");
            addOption("feature-type", "Type of the feature (class or score)");

            options.addOption("mh", false, "Metropolis-Hastings");
            options.addOption("train", false, "train");
            options.addOption("dev", false, "development");
            options.addOption("test", false, "test");
            options.addOption("parallel", false, "parallel");
            options.addOption("display", false, "display");
            options.addOption("hack", false, "hack");
            options.addOption("paramOpt", false, "Optimizing parameters");
            options.addOption("diagnose", false, "diagnose");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("z", false, "z-normalize");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);

            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(VotePredExpt.class
                        .getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");
            verbose = true;
            debug = true;

            Congress.setVerbose(verbose);
            Congress.setDebug(debug);
            TextDataset.setDebug(debug);
            TextDataset.setVerbose(verbose);

            BillTextExperiment expt = new BillTextExperiment();
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
            switch (runMode) {
                case "preprocess":
                    expt.preprocess();
                    break;
                case "run":
                    expt.run();
                    break;
                default:
                    throw new RuntimeException("Run mode " + runMode + " is not supported");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
