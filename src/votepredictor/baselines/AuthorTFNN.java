package votepredictor.baselines;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import util.IOUtils;
import util.RankingItem;
import util.SparseVector;
import votepredictor.AbstractVotePredictor;

/**
 * Make predictions of votes by unknown voters on a set of known votes using the
 * voters' text. Each voter is represented by a term-frequency (TF) vector.
 *
 * @author vietan
 */
public class AuthorTFNN extends AbstractVotePredictor {

    // input
    protected int V;
    protected ArrayList<Integer> billIndices;
    protected int B;

    protected SparseVector[] authorVectors;                 // A
    protected HashMap<Integer, Integer>[] authorVoteMap;    // A

    public AuthorTFNN(String bname) {
        super(bname);
    }

    @Override
    public String getName() {
        return this.name;
    }

    public void configure(int V) {
        this.V = V;
    }

    /**
     * Set training data.
     *
     * @param docIndices Indices of selected documents
     * @param words Document words
     * @param authors Document authors
     * @param votes All votes
     * @param authorIndices Indices of selected authors
     * @param billIndices Indices of selected bills
     * @param trainVotes Training votes
     */
    public void train(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] trainVotes) {
        if (verbose) {
            logln("Setting up training ...");
        }
        // list of training authors
        if (authorIndices == null) {
            authorIndices = new ArrayList<>();
            for (int aa = 0; aa < votes.length; aa++) {
                authorIndices.add(aa);
            }
        }
        int A = authorIndices.size();

        // list of bills
        this.billIndices = billIndices;
        if (billIndices == null) {
            this.billIndices = new ArrayList<>();
            for (int bb = 0; bb < votes[0].length; bb++) {
                this.billIndices.add(bb);
            }
        }
        this.B = this.billIndices.size();

        // training votes
        this.authorVoteMap = new HashMap[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorVoteMap[aa] = new HashMap<>();
            int author = authorIndices.get(aa);
            for (int bb : this.billIndices) {
                if (trainVotes[author][bb]) {
                    this.authorVoteMap[aa].put(bb, votes[author][bb]);
                }
            }
        }

        if (docIndices == null) { // add all documents
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                docIndices.add(dd);
            }
        }

        this.authorVectors = new SparseVector[A];
        for (int aa = 0; aa < A; aa++) {
            this.authorVectors[aa] = new SparseVector(V);
        }
        for (int dd : docIndices) {
            int author = authors[dd];
            int aa = authorIndices.indexOf(author);
            if (aa < 0) {
                continue;
            }
            for (int nn = 0; nn < words[dd].length; nn++) {
                this.authorVectors[aa].change(words[dd][nn], 1.0);
            }
        }
        for (SparseVector authorVec : this.authorVectors) {
            authorVec.normalize();
        }
    }

    public SparseVector[] test(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            ArrayList<Integer> authorIndices,
            boolean[][] testVotes,
            int K,
            int[][] votes) {
        if (authorIndices == null) {
            throw new RuntimeException("List of test authors is null");
        }
        int testA = authorIndices.size();
        SparseVector[] testAuthorVecs = new SparseVector[testA];
        for (int aa = 0; aa < testA; aa++) {
            testAuthorVecs[aa] = new SparseVector(this.V);
        }
        if (docIndices == null) { // add all documents
            docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                docIndices.add(dd);
            }
        }
        for (int dd : docIndices) {
            int author = authors[dd];
            int aa = authorIndices.indexOf(author);
            if (aa < 0) {
                continue;
            }
            for (int nn = 0; nn < words[dd].length; nn++) {
                testAuthorVecs[aa].change(words[dd][nn], 1.0);
            }
        }

        for (SparseVector testAuthorVec : testAuthorVecs) {
            testAuthorVec.normalize();
        }

        SparseVector[] predictions = new SparseVector[testVotes.length];
        for (int aa = 0; aa < testA; aa++) {
            int author = authorIndices.get(aa);

            // get nearest neighbors
            ArrayList<RankingItem<Integer>> rankNeighbors = new ArrayList<>();
            for (int ii = 0; ii < this.authorVectors.length; ii++) {
                double sim = this.authorVectors[ii].cosineSimilarity(testAuthorVecs[aa]);
                rankNeighbors.add(new RankingItem<Integer>(ii, sim));
            }
            Collections.sort(rankNeighbors);

            predictions[author] = new SparseVector(this.B);
            for (int bb = 0; bb < this.B; bb++) {
                if (testVotes[author][bb]) {
                    double num = 0.0;
                    double den = 0.0;
                    for (int kk = 0; kk < K; kk++) {
                        RankingItem<Integer> rankNeighbor = rankNeighbors.get(kk);
                        int neighbor = rankNeighbor.getObject();
                        Integer vote = authorVoteMap[neighbor].get(bb);
                        if (vote != null) {
                            double sim = rankNeighbor.getPrimaryValue();
                            num += sim * vote;
                            den += sim;
                        }
                    }
                    double val = num / den;
                    predictions[author].set(bb, val);
                }
            }
        }

        return predictions;
    }

    @Override
    public void output(File outputFile) {
        if (verbose) {
            logln("Outputing model to " + outputFile);
        }

        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            writer.write(Integer.toString(V) + "\n");
            // output learned authors' vectors
            writer.write(Integer.toString(authorVectors.length) + "\n");
            for (int aa = 0; aa < this.authorVectors.length; aa++) {
                writer.write(aa + "\t" + SparseVector.output(authorVectors[aa]) + "\n");
            }
            // output bills
            writer.write(Integer.toString(B) + "\n");
            for (int bb : billIndices) {
                writer.write(bb + "\n");
            }
            // output votes
            for (int aa = 0; aa < authorVoteMap.length; aa++) {
                writer.write(Integer.toString(aa));
                for (int bb : authorVoteMap[aa].keySet()) {
                    writer.write("\t" + bb + ":" + authorVoteMap[aa].get(bb));
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing to " + outputFile);
        }
    }

    @Override
    public void input(File inputFile) {
        if (verbose) {
            logln("Inputing model from " + inputFile);
        }
        try {
            BufferedReader reader = IOUtils.getBufferedReader(inputFile);
            this.V = Integer.parseInt(reader.readLine());
            // input author vectors
            int A = Integer.parseInt(reader.readLine());
            this.authorVectors = new SparseVector[A];
            for (int aa = 0; aa < A; aa++) {
                String[] sline = reader.readLine().split("\t");
                int authorIdx = Integer.parseInt(sline[0]);
                if (aa != authorIdx) {
                    throw new RuntimeException("MISMATCH");
                }
                if (sline.length == 1) {
                    this.authorVectors[aa] = new SparseVector(V);
                } else {
                    this.authorVectors[aa] = SparseVector.input(sline[1]);
                }
            }
            // input bills
            this.B = Integer.parseInt(reader.readLine());
            this.billIndices = new ArrayList<>();
            for (int ii = 0; ii < B; ii++) {
                this.billIndices.add(Integer.parseInt(reader.readLine()));
            }
            // input votes
            this.authorVoteMap = new HashMap[A];
            for (int aa = 0; aa < A; aa++) {
                String[] sline = reader.readLine().split("\t");
                if (Integer.parseInt(sline[0]) != aa) {
                    throw new RuntimeException("Mismatch");
                }
                this.authorVoteMap[aa] = new HashMap<>();
                for (int ii = 1; ii < sline.length; ii++) {
                    int bb = Integer.parseInt(sline[ii].split(":")[0]);
                    int val = Integer.parseInt(sline[ii].split(":")[1]);
                    this.authorVoteMap[aa].put(bb, val);
                }
            }
            reader.close();
        } catch (IOException | RuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while inputing from " + inputFile);
        }
    }
}
