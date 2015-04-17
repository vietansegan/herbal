package votepredictor.textidealpoint;

import core.AbstractSampler;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import sampler.unsupervised.LDA;
import util.IOUtils;
import util.MiscUtils;

/**
 *
 * @author vietan
 */
public abstract class AbstractTextIdealPoint extends AbstractSampler {

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
    // configure
    protected WordWeightType wordWeightType;
    protected double[] wordWeights;
    protected double[] authorTotalWordWeights;

    protected ArrayList<String> authorVocab;
    protected ArrayList<String> voteVocab;

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public ArrayList<String> getAuthorVocab() {
        return this.authorVocab;
    }

    public void setVoteVocab(ArrayList<String> voteVoc) {
        this.voteVocab = voteVoc;
    }

    public void setVotes(int[][] v) {
        this.votes = v;
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
     * Run LDA.
     *
     * @param K Number of topics
     */
    public LDA runLDA(int K) {
        int lda_burnin = 250;
        int lda_maxiter = 500;
        int lda_samplelag = 25;
        LDA lda = new LDA();
        lda.setDebug(false);
        lda.setVerbose(verbose);
        lda.setLog(false);
        double lda_alpha = 0.1;
        double lda_beta = 0.1;

        lda.configure(folder, V, K, lda_alpha, lda_beta, InitialState.RANDOM, false,
                lda_burnin, lda_maxiter, lda_samplelag, lda_samplelag);

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
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while running LDA for initialization");
        }
        setLog(log);
        return lda;
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
            logln("Setting up data ...");
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

        // update author vocab
        if (authorVocab != null) {
            ArrayList<String> aVoc = new ArrayList<>();
            for (int aa = 0; aa < A; aa++) {
                aVoc.add(authorVocab.get(this.authorIndices.get(aa)));
            }
            this.authorVocab = aVoc;
        }

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

        // TODO: update vote vocab
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

        // skip voters/bills which don't have any vote
        this.validAs = new boolean[A];
        this.validBs = new boolean[B];
        for (int aa = 0; aa < A; aa++) {
            for (int bb = 0; bb < B; bb++) {
                if (isValidVote(aa, bb)) {
                    this.validAs[aa] = true;
                    this.validBs[bb] = true;
                }
            }
        }

        this.setWordWeightType();

        this.prepareDataStatistics();

        if (verbose) {
            logln("--- # documents:\t" + D);
            logln("--- # voters:\t" + A);
            logln("--- # bills:\t" + B);
            logln("--- # tokens:\t" + numTokens);
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
