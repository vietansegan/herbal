package votepredictor;

import core.AbstractSampler;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public abstract class AbstractTopicBasedIdealPoint extends AbstractSampler {

    public static final String AuthorFileExt = ".author";
    public static final String BillFileExt = ".bill";

    public enum WordWeightType {

        NONE, TFIDF
    }

    // inputs
    protected int[][] words;
    protected ArrayList<Integer> docIndices;
    protected ArrayList<Integer> authorIndices;
    protected ArrayList<Integer> billIndices;
    protected int[] authors; // [D]: author of each document
    protected int[][] votes;
    protected boolean[][] validVotes;
    protected int V; // vocabulary size
    // derive
    protected int D; // number of documents
    protected int A; // number of authors
    protected int B; // number of bills
    protected boolean[] validAs; // flag voters with no training vote
    protected boolean[] validBs; // flag bills with no training vote
    protected int numTokens;
    // configure
    protected WordWeightType wordWeightType;
    protected double[] wordWeights;
    protected double[] authorTotalWordWeights;

    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public void setVoteVocab(ArrayList<String> voteVoc) {
        this.voteVocab = voteVoc;
    }

    public int getVote(int aa, int bb) {
        return this.votes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    public boolean isValidVote(int aa, int bb) {
        return this.validVotes[this.authorIndices.get(aa)][this.billIndices.get(bb)];
    }

    protected void setTestConfigurations() {
        setTestConfigurations(100, 250, 10, 5);
    }

    public void setWordWeightType() {
        if (this.wordWeightType == WordWeightType.NONE) {
            this.wordWeights = new double[V];
            Arrays.fill(this.wordWeights, 1.0);
        } else if (this.wordWeightType == WordWeightType.TFIDF) {
            this.wordWeights = MiscUtils.getIDFs(words, V);
        } else {
            throw new RuntimeException("WordWeightType " + this.wordWeightType
                    + " not supported");
        }
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
    public void setupData(ArrayList<Integer> docIndices,
            int[][] words,
            int[] authors,
            int[][] votes,
            ArrayList<Integer> authorIndices,
            ArrayList<Integer> billIndices,
            boolean[][] trainVotes) {
        if (verbose) {
            logln("Setting up training ...");
        }
        // list of authors
        this.authorIndices = authorIndices;
        if (authorIndices == null) {
            this.authorIndices = new ArrayList<>();
            for (int aa = 0; aa < trainVotes.length; aa++) {
                this.authorIndices.add(aa);
            }
        }
        this.A = this.authorIndices.size();

        HashMap<Integer, Integer> inverseAuthorMap = new HashMap<>();
        for (int ii = 0; ii < A; ii++) {
            int aa = this.authorIndices.get(ii);
            inverseAuthorMap.put(aa, ii);
        }

        // list of bills
        this.billIndices = billIndices;
        if (billIndices == null) {
            this.billIndices = new ArrayList<>();
            for (int bb = 0; bb < trainVotes[0].length; bb++) {
                this.billIndices.add(bb);
            }
        }
        this.B = this.billIndices.size();

        this.votes = votes;
        this.validVotes = trainVotes;

        this.docIndices = docIndices;
        if (this.docIndices == null) { // add all documents
            this.docIndices = new ArrayList<>();
            for (int dd = 0; dd < words.length; dd++) {
                this.docIndices.add(dd);
            }
        }
        this.D = this.docIndices.size();
        this.words = new int[D][];
        this.authors = new int[D];
        for (int ii = 0; ii < this.D; ii++) {
            int dd = this.docIndices.get(ii);
            this.words[ii] = words[dd];
            this.authors[ii] = inverseAuthorMap.get(authors[dd]);
        }

        this.setWordWeightType();

        this.prepareDataStatistics();

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # voters:\t" + A);
            logln("--- # bills:\t" + B);
        }
    }

    /**
     * Pre-computed statistics.
     */
    protected void prepareDataStatistics() {
        // statistics
        numTokens = 0;
        for (int d = 0; d < D; d++) {
            numTokens += this.words[d].length;
        }

        // author document list
        ArrayList<Integer>[] authorDocList = new ArrayList[A];
        for (int a = 0; a < A; a++) {
            authorDocList[a] = new ArrayList<Integer>();
        }
        for (int d = 0; d < D; d++) {
            if (this.words[d].length > 0) {
                authorDocList[authors[d]].add(d);
            }
        }
        this.authorTotalWordWeights = new double[A];
        for (int dd = 0; dd < D; dd++) {
            int aa = authors[dd];
            for (int nn = 0; nn < words[dd].length; nn++) {
                this.authorTotalWordWeights[aa] += wordWeights[words[dd][nn]];
            }
        }
    }
}
