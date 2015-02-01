package votepredictor;

import core.AbstractModel;
import data.Vote;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import util.IOUtils;
import util.SparseVector;
import util.evaluation.Measurement;
import util.evaluation.RankingEvaluation;

/**
 *
 * @author vietan
 */
public abstract class AbstractVotePredictor extends AbstractModel {

    public static final String AuthorScoreFile = "authors.score";
    public static final String VoteScoreFile = "votes.score";

    public AbstractVotePredictor() {
    }

    public AbstractVotePredictor(String name) {
        super(name);
    }

    /**
     * Output author scores.
     *
     * @param outputFile Output file
     * @param authors
     * @param authorScores
     */
    public static void outputAuthorScores(File outputFile,
            String[] authors, double[] authorScores) {
        logln("Outputing author scores to " + outputFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Author\tScore\n");
            for (int aa = 0; aa < authorScores.length; aa++) {
                if (authors != null) {
                    writer.write(authors[aa] + "\t" + authorScores[aa] + "\n");
                } else {
                    writer.write(aa + "\t" + authorScores[aa] + "\n");
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing author scores "
                    + "to " + outputFile);
        }
    }

    /**
     * Input author scores.
     *
     * @param inputFile
     * @return
     */
    public static double[] inputAuthorScores(File inputFile) {
        logln("Inputing author scores from " + inputFile);
        double[] authorScores = null;
        try {
            ArrayList<Double> scoreList = new ArrayList<>();
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            reader.readLine(); // header
            String line;
            while ((line = reader.readLine()) != null) {
                double score = Double.parseDouble(line.split("\t")[1]);
                scoreList.add(score);
            }
            reader.close();

            authorScores = new double[scoreList.size()];
            for (int ii = 0; ii < authorScores.length; ii++) {
                authorScores[ii] = scoreList.get(ii);
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing author scores "
                    + "from " + inputFile);
        }
        return authorScores;
    }

    /**
     * Output vote scores.
     *
     * @param outputFile Output file
     * @param votes
     * @param voteXs
     * @param voteYs
     */
    public static void outputVoteScores(File outputFile, String[] votes,
            double[] voteXs, double[] voteYs) {
        logln("Outputing vote scores to " + outputFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write("Vote\tScore\n");
            for (int bb = 0; bb < voteXs.length; bb++) {
                if (votes != null) {
                    writer.write(votes[bb] + "\t" + voteXs[bb] + "\t" + voteYs[bb] + "\n");
                } else {
                    writer.write(bb + "\t" + voteXs[bb] + "\t" + voteYs[bb] + "\n");
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing vote scores "
                    + "to " + outputFile);
        }
    }

    /**
     * Input vote scores.
     *
     * @param inputFile
     * @return
     */
    public static double[][] inputVoteScores(File inputFile) {
        logln("Inputing vote scores from " + inputFile);
        double[][] voteScores = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            reader.readLine();
            String line;
            ArrayList<Double> voteXs = new ArrayList<>();
            ArrayList<Double> voteYs = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                String[] sline = line.split("\t");
                voteXs.add(Double.parseDouble(sline[1]));
                voteYs.add(Double.parseDouble(sline[2]));
            }
            reader.close();

            voteScores = new double[voteXs.size()][2];
            for (int ii = 0; ii < voteScores.length; ii++) {
                voteScores[ii][0] = voteXs.get(ii);
                voteScores[ii][1] = voteYs.get(ii);
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing vote scores "
                    + "from " + inputFile);
        }
        return voteScores;
    }

    /**
     * Output the predicted scores for each test vote.
     *
     * @param predFile Output file
     * @param votes True votes (optional)
     * @param predictions Each sparse vector corresponds to a voter. The key of
     * the sparse vector is the vote index and the value is the predicted
     * probability that the corresponding vote is 1.
     */
    public static void outputPredictions(File predFile, int[][] votes,
            SparseVector[] predictions) {
        System.out.println("Outputing predictions to " + predFile);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(predFile);
            writer.write(predictions.length + "\n");
            for (int aa = 0; aa < predictions.length; aa++) {
                if (predictions[aa] == null || predictions[aa].isEmpty()) {
                    continue;
                }
                writer.write(Integer.toString(aa));
                for (int key : predictions[aa].getIndices()) {
                    writer.write("\t" + key
                            + ":" + predictions[aa].get(key));
                    if (votes != null) {
                        writer.write(":" + votes[aa][key]);
                    }
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing predictions to "
                    + predFile);
        }
    }

    /**
     * Input vote predictions.
     *
     * @param predFile File storing predictions
     * @return Prediction
     */
    public static SparseVector[] inputPredictions(File predFile) {
        SparseVector[] predictions = null;
        try {
            BufferedReader reader = IOUtils.getBufferedReader(predFile);
            int numPreds = Integer.parseInt(reader.readLine());
            predictions = new SparseVector[numPreds];
            for (int ii = 0; ii < numPreds; ii++) {
                predictions[ii] = new SparseVector();
            }
            String line;
            while ((line = reader.readLine()) != null) {
                String[] sline = line.split("\t");
                int aa = Integer.parseInt(sline[0]);
                for (int jj = 1; jj < sline.length; jj++) {
                    int key = Integer.parseInt(sline[jj].split(":")[0]);
                    double pred = Double.parseDouble(sline[jj].split(":")[1]);
                    predictions[aa].set(key, pred);
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing predictions from "
                    + predFile);
        }
        return predictions;
    }

    /**
     * Evaluate predicted values for each vote.
     *
     * @param votes Binary matrix storing votes (some votes can be missing)
     * @param testVotes Boolean matrix indicating valid votes
     * @param predictedValues Predicted values
     * @return List of evaluation measurements
     */
    public static ArrayList<Measurement> evaluateAll(
            int[][] votes,
            boolean[][] testVotes,
            SparseVector[] predictedValues) {
        ArrayList<Measurement> measurements = new ArrayList<>();

        double llh = 0.0;
        int count = 0;
        int posCount = 0;
        int negCount = 0;
        int correctCount = 0;
        double mae = 0.0;
        double mse = 0.0;
        Set<String> withVotes = new HashSet<>();
        ArrayList<String> voteList = new ArrayList<>();
        ArrayList<Double> voteScores = new ArrayList<>();
        for (int aa = 0; aa < votes.length; aa++) {
            for (int vv = 0; vv < votes[aa].length; vv++) {
                if (!testVotes[aa][vv]) {
                    continue;
                }
                String key = aa + "_" + vv;
                double val = predictedValues[aa].get(vv);
                voteList.add(key);
                voteScores.add(val);
                mae += Math.abs(votes[aa][vv] - val);
                mse += Math.pow(votes[aa][vv] - val, 2);

                if (votes[aa][vv] == Vote.WITH) {
                    withVotes.add(key);
                    llh += Math.log(val);
                    if (val >= 0.5) {
                        correctCount++;
                    }
                    posCount++;
                } else if (votes[aa][vv] == Vote.AGAINST) {
                    llh += Math.log(1.0 - val);
                    if (val < 0.5) {
                        correctCount++;
                    }
                    negCount++;
                } else {
                    throw new RuntimeException("Missing data");
                }
                count++;
            }
        }

        // compute log likelihood of test data
        measurements.add(new Measurement("count", count));
        measurements.add(new Measurement("positive count", posCount));
        measurements.add(new Measurement("negative count", negCount));
        measurements.add(new Measurement("loglikelihood", llh));
        measurements.add(new Measurement("avg-loglikelihood", llh / count));
        measurements.add(new Measurement("accuracy", (double) correctCount / count));
        measurements.add(new Measurement("mae", (double) mae / count));
        measurements.add(new Measurement("mse", (double) mse / count));

        // compute ranking performances
        double[] scores = new double[voteScores.size()];
        Set<Integer> withIndices = new HashSet<>();
        for (int ii = 0; ii < scores.length; ii++) {
            scores[ii] = voteScores.get(ii);

            String vote = voteList.get(ii);
            if (withVotes.contains(vote)) {
                withIndices.add(ii);
            }
        }
        RankingEvaluation withRankPerf = new RankingEvaluation(scores, withIndices);
        withRankPerf.computeAUCs();
//        withRankPerf.computePRF();
        for (Measurement m : withRankPerf.getMeasurements()) {
            measurements.add(m);
        }
        return measurements;
    }
    
    public static ArrayList<Measurement> evaluate(
            int[][] votes,
            boolean[][] testVotes,
            SparseVector[] predictedValues) {
        ArrayList<Measurement> measurements = new ArrayList<>();

        double llh = 0.0;
        int count = 0;
        int posCount = 0;
        int negCount = 0;
        int correctCount = 0;
        double mae = 0.0;
        double mse = 0.0;
        Set<String> withVotes = new HashSet<>();
        ArrayList<String> voteList = new ArrayList<>();
        ArrayList<Double> voteScores = new ArrayList<>();
        for (int aa = 0; aa < votes.length; aa++) {
            for (int vv = 0; vv < votes[aa].length; vv++) {
                if (!testVotes[aa][vv]) {
                    continue;
                }
                String key = aa + "_" + vv;
                double val = predictedValues[aa].get(vv);
                voteList.add(key);
                voteScores.add(val);
                mae += Math.abs(votes[aa][vv] - val);
                mse += Math.pow(votes[aa][vv] - val, 2);

                if (votes[aa][vv] == Vote.WITH) {
                    withVotes.add(key);
                    llh += Math.log(val);
                    if (val >= 0.5) {
                        correctCount++;
                    }
                    posCount++;
                } else if (votes[aa][vv] == Vote.AGAINST) {
                    llh += Math.log(1.0 - val);
                    if (val < 0.5) {
                        correctCount++;
                    }
                    negCount++;
                } else {
                    throw new RuntimeException("Missing data");
                }
                count++;
            }
        }

        // compute log likelihood of test data
        measurements.add(new Measurement("count", count));
        measurements.add(new Measurement("positive count", posCount));
        measurements.add(new Measurement("negative count", negCount));
        measurements.add(new Measurement("loglikelihood", llh));
        measurements.add(new Measurement("avg-loglikelihood", llh / count));
        measurements.add(new Measurement("accuracy", (double) correctCount / count));
        measurements.add(new Measurement("mae", (double) mae / count));
        measurements.add(new Measurement("mse", (double) mse / count));
        return measurements;
    }
}
