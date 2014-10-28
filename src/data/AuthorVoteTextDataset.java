package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import sampling.util.SparseCount;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class AuthorVoteTextDataset extends TextDataset {

    public static final String voteVocabExt = ".vvoc";
    public static final String authorVoteExt = ".votes";
    public static final String voteTextExt = ".votetext";
    public static final int AGAINST = 0;
    public static final int WITH = 1;
    // header in author vocab file
    protected ArrayList<String> authorProperties;
    // raw author list
    protected ArrayList<String> authorList;
    // raw author vote
    protected HashMap<String, HashMap<Integer, Integer>> rawAuthorVotes;
    // processed authors
    protected ArrayList<String> authorVocab;
    protected int[] authors;
    protected HashMap<String, Author> authorTable;
    // votes
    protected ArrayList<String> voteVocab;
    protected ArrayList<String> voteText;
    private int[][] voteWords;
    // author specific data
    protected int[][] authorWords;
    protected int[][][] authorSentWords;
    protected String[][] authorRawSents;
    protected HashMap<Integer, Integer>[] authorVotes; // length = # authors
    protected int[][] votes;
    protected HashMap<String, Vote> voteTable;
    protected ArrayList<String> voteProperties;

    public AuthorVoteTextDataset(String name) {
        super(name);
        this.authorTable = new HashMap<String, Author>();
        this.voteTable = new HashMap<String, Vote>();
    }

    public AuthorVoteTextDataset(String name, String folder) {
        super(name, folder);
        this.authorTable = new HashMap<String, Author>();
        this.voteTable = new HashMap<String, Vote>();
    }

    public AuthorVoteTextDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
        this.authorTable = new HashMap<String, Author>();
        this.voteTable = new HashMap<String, Vote>();
    }

    public HashMap<String, Author> getAuthorTable() {
        return this.authorTable;
    }

    public HashMap<String, Vote> getVoteTable() {
        return this.voteTable;
    }

    public void formatAuthorData() {
        this.formatAuthorRawSentences();
        this.formatAuthorSentWords();
        this.formatAuthorWords();
    }

    public int[][] getVotes() {
        return this.votes;
    }

    public void setVotePropertyNames(ArrayList<String> voteProps) {
        this.voteProperties = voteProps;
    }

    public void setAuthorPropertyNames(ArrayList<String> authorProps) {
        this.authorProperties = authorProps;
    }

    public void setAuthorVotes(HashMap<String, HashMap<Integer, Integer>> aVotes) {
        this.rawAuthorVotes = aVotes;
    }

    public HashMap<Integer, Integer>[] getAuthorVotes() {
        return this.authorVotes;
    }

    public void setVoteVocab(ArrayList<String> vVoc) {
        this.voteVocab = vVoc;
    }

    public ArrayList<String> getVoteVocab() {
        return this.voteVocab;
    }

    public void setVoteText(ArrayList<String> vText) {
        this.voteText = vText;
    }

    public ArrayList<String> getAuthorVocab() {
        return this.authorVocab;
    }

    public void setAuthorVocab(ArrayList<String> authorVoc) {
        this.authorVocab = authorVoc;
    }

    public int[] getAuthors() {
        return this.authors;
    }

    public void setVoteProperty(String voteId, String propName, String propVal) {
        Vote vote = voteTable.get(voteId);
        if (vote == null) {
            vote = new Vote(voteId);
        }
        vote.addProperty(propName, propVal);
        voteTable.put(voteId, vote);
    }

    public void setAuthorProperty(String authorId, String propName, String propVal) {
        Author author = authorTable.get(authorId);
        if (author == null) {
            author = new Author(authorId);
        }
        author.addProperty(propName, propVal);
        authorTable.put(authorId, author);
    }

    public String getVoteProperty(String voteId, String propName) {
        Vote vote = voteTable.get(voteId);
        if (vote == null) {
            return null;
        }
        return vote.getProperty(propName);
    }

    public String getAuthorProperty(String authorId, String propName) {
        Author author = authorTable.get(authorId);
        if (author == null) {
            return null;
        }
        return author.getProperty(propName);
    }

    public void setAuthorList(ArrayList<String> authorList) {
        this.authorList = authorList;
    }

    public String[] getAuthorIds() {
        return this.authorVocab.toArray(new String[authorVocab.size()]);
    }

    public String[][] getAuthorRawSentences() {
        return this.authorRawSents;
    }

    public int[][][] getAuthorSentWords() {
        return this.authorSentWords;
    }

    public int[][] getAuthorWords() {
        return this.authorWords;
    }

    private void formatAuthorRawSentences() {
        HashMap<Integer, ArrayList<String>> authorRawSentMap
                = new HashMap<Integer, ArrayList<String>>();
        for (int d = 0; d < sentRawWords.length; d++) {
            int author = authors[d];
            ArrayList<String> authorRawSentList = authorRawSentMap.get(author);
            if (authorRawSentList == null) {
                authorRawSentList = new ArrayList<String>();
            }
            authorRawSentList.addAll(Arrays.asList(sentRawWords[d]));
            authorRawSentMap.put(author, authorRawSentList);
        }

        authorRawSents = new String[authorVocab.size()][];
        for (int a = 0; a < authorVocab.size(); a++) {
            ArrayList<String> authorSents = authorRawSentMap.get(a);
            if (authorRawSents == null) {
                authorRawSents[a] = new String[0];
                continue;
            }
            authorRawSents[a] = new String[authorSents.size()];
            for (int ii = 0; ii < authorSents.size(); ii++) {
                authorRawSents[a][ii] = authorSents.get(ii);
            }
        }
    }

    private void formatAuthorSentWords() {
        HashMap<Integer, ArrayList<int[]>> authorSentWordListMap
                = new HashMap<Integer, ArrayList<int[]>>();
        for (int d = 0; d < sentWords.length; d++) {
            int author = authors[d];
            ArrayList<int[]> authorSentWordList = authorSentWordListMap.get(author);
            if (authorSentWordList == null) {
                authorSentWordList = new ArrayList<int[]>();
            }
            authorSentWordList.addAll(Arrays.asList(sentWords[d]));
            authorSentWordListMap.put(author, authorSentWordList);
        }

        authorSentWords = new int[authorVocab.size()][][];
        for (int a = 0; a < authorVocab.size(); a++) {
            ArrayList<int[]> authorSentWordList = authorSentWordListMap.get(a);
            if (authorSentWordList == null) {
                authorSentWords[a] = new int[0][];
                continue;
            }
            int[][] sws = new int[authorSentWordList.size()][];
            for (int ii = 0; ii < sws.length; ii++) {
                sws[ii] = authorSentWordList.get(ii);
            }
            authorSentWords[a] = sws;
        }
    }

    public void formatAuthorWords() {
        HashMap<Integer, ArrayList<Integer>> authorWordListMap = new HashMap<Integer, ArrayList<Integer>>();
        for (int d = 0; d < words.length; d++) {
            int author = authors[d];
            ArrayList<Integer> authorWordList = authorWordListMap.get(author);
            if (authorWordList == null) {
                authorWordList = new ArrayList<Integer>();
            }

            for (int n = 0; n < words[d].length; n++) {
                authorWordList.add(words[d][n]);
            }
            authorWordListMap.put(author, authorWordList);
        }

        authorWords = new int[authorVocab.size()][];
        for (int a = 0; a < authorWords.length; a++) {
            ArrayList<Integer> authorWordList = authorWordListMap.get(a);
            if (authorWordList == null) {
                authorWords[a] = new int[0];
                continue;
            }
            int[] ws = new int[authorWordList.size()];
            for (int ii = 0; ii < ws.length; ii++) {
                ws[ii] = authorWordList.get(ii);
            }
            authorWords[a] = ws;
        }
    }

    @Override
    public void format(File outputFolder) throws Exception {
        format(outputFolder.getAbsolutePath());
    }

    @Override
    public void format(String outputFolder) throws Exception {
        IOUtils.createFolder(outputFolder);

        String[] rawTexts = textList.toArray(new String[textList.size()]);
        corpProc.setRawTexts(rawTexts);
        corpProc.process();

        if (voteText != null) {
            formatVoteText(outputFolder);
        }

        outputWordVocab(outputFolder);
        outputTextData(outputFolder);

        formatAuthors(outputFolder);

        outputDocumentInfo(outputFolder);
        outputVoteVocab(outputFolder);
        outputAuthorVotes(outputFolder);
        if (sent) {
            outputSentTextData(outputFolder);
        }
    }

    private void formatVoteText(String outputFolder) {
        if (verbose) {
            logln("--- Formatting vote text ...");
        }
        CorpusProcessor cp = new CorpusProcessor(corpProc);
        cp.unigramCountCutoff = 1;
        cp.bigramCountCutoff = 2;
        cp.docTypeCountCutoff = 2;
        String[] rawTexts = voteText.toArray(new String[voteText.size()]);
        cp.setRawTexts(rawTexts);
        cp.process();

        ArrayList<String> curWordVocab = corpProc.getVocab();
        ArrayList<String> newWordVocab = cp.getVocab();
        if (verbose) {
            logln("--- --- Word vocab size (before): " + curWordVocab.size());
            logln("--- --- Vote Word vocab size: " + newWordVocab.size());
        }

        HashMap<Integer, Integer> mapping = new HashMap<>();
        for (int vv = 0; vv < newWordVocab.size(); vv++) {
            String word = newWordVocab.get(vv);
            int idx = Collections.binarySearch(curWordVocab, word);
            if (idx < 0) {
                mapping.put(vv, curWordVocab.size());
                curWordVocab.add(word);
            } else {
                mapping.put(vv, idx);
            }
        }
        if (verbose) {
            logln("--- --- Word vocab size (after): " + curWordVocab.size());
        }

        corpProc.setVocab(curWordVocab);

        // update indexed words
        voteWords = cp.getNumerics();
        for (int[] voteWord : voteWords) {
            for (int nn = 0; nn < voteWord.length; nn++) {
                Integer mappedWord = mapping.get(voteWord[nn]);
                if (mappedWord == null) {
                    throw new RuntimeException("NULL. " + voteWord[nn]);
                }
                voteWord[nn] = mappedWord;
            }
        }

        File voteTextFile = new File(outputFolder, formatFilename + voteTextExt);
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(voteTextFile);
            for (int[] voteWord : voteWords) {
                SparseCount typeCounts = new SparseCount();
                for (int jj = 0; jj < voteWord.length; jj++) {
                    typeCounts.increment(voteWord[jj]);
                }
                writer.write(Integer.toString(typeCounts.getIndices().size()));
                for (int vv : typeCounts.getSortedIndices()) {
                    writer.write(" " + vv + ":" + typeCounts.getCount(vv));
                }
                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing vote text to "
                    + voteTextFile);
        }
    }

    protected void inputVoteTextData(File file) throws Exception {
        if (verbose) {
            logln("--- Reading vote text data from " + file);
        }

        voteWords = inputFormattedTextData(file);

        if (verbose) {
            logln("--- --- # docs: " + voteWords.length);
            int numTokens = 0;
            for (int[] word : voteWords) {
                numTokens += word.length;
            }
            logln("--- --- # tokens: " + numTokens);
        }
    }

    @Override
    protected void outputTextData(String outputFolder) throws Exception {
        File outputFile = new File(outputFolder, formatFilename + numDocDataExt);
        if (verbose) {
            logln("--- Outputing main numeric data ... " + outputFile);
        }

        // output main numeric
        int[][] numDocs = corpProc.getNumerics();
        BufferedWriter dataWriter = IOUtils.getBufferedWriter(outputFile);
        for (int d = 0; d < numDocs.length; d++) {
            HashMap<Integer, Integer> typeCounts = new HashMap<Integer, Integer>();
            for (int j = 0; j < numDocs[d].length; j++) {
                Integer count = typeCounts.get(numDocs[d][j]);
                if (count == null) {
                    typeCounts.put(numDocs[d][j], 1);
                } else {
                    typeCounts.put(numDocs[d][j], count + 1);
                }
            }

            // skip short documents
            if (typeCounts.size() < corpProc.docTypeCountCutoff) {
                continue;
            }

            String author = authorList.get(d);
            if (this.rawAuthorVotes.get(author) == null) { // skip author without votes
                continue;
            }

            // write main data
            dataWriter.write(Integer.toString(typeCounts.size()));
            for (int type : typeCounts.keySet()) {
                dataWriter.write(" " + type + ":" + typeCounts.get(type));
            }
            dataWriter.write("\n");

            // save the doc id
            this.processedDocIndices.add(d);
        }
        dataWriter.close();
    }

    protected void formatAuthors(String outputFolder) throws Exception {
        if (verbose) {
            logln("--- Formatting authors. " + outputFolder);
        }
        // create author vocab
        if (this.authorVocab == null) { // create author vocab if not exists
            this.authorVocab = new ArrayList<String>();
            for (int docIndex : this.processedDocIndices) {
                String author = authorList.get(docIndex);
                if (this.rawAuthorVotes.get(author) == null) { // skip author without votes
                    continue;
                }

                if (!this.authorVocab.contains(author)) {
                    this.authorVocab.add(author);
                }
            }
        }

        if (verbose) {
            logln("--- --- Author vocab size: " + this.authorVocab.size());
        }

        // output author vocab
        BufferedWriter writer = IOUtils.getBufferedWriter(new File(outputFolder,
                formatFilename + speakerVocabExt));
        // - headers
        writer.write("ID");
        for (String prop : authorProperties) {
            writer.write("\t" + prop);
        }
        writer.write("\n");

        // - authors
        for (String authorId : this.authorVocab) {
            Author author = authorTable.get(authorId);
            writer.write(authorId);
            for (String property : this.authorProperties) {
                writer.write("\t" + author.getProperty(property));
            }
            writer.append("\n");
        }
        writer.close();
    }

    protected void outputVoteVocab(String outputFolder) throws Exception {
        File voteVocFile = new File(outputFolder, formatFilename + voteVocabExt);
        if (verbose) {
            logln("--- Outputing vote vocab ..." + voteVocFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(voteVocFile);
        writer.write("ID");
        for (String prop : voteProperties) {
            writer.write("\t" + prop);
        }
        writer.write("\n");

        for (String vid : voteVocab) {
            writer.write(vid);
            for (String prop : voteProperties) {
                writer.write("\t" + getVoteProperty(vid, prop));
            }
            writer.write("\n");
        }
        writer.close();
    }

    protected void inputVoteVocab(File file) throws Exception {
        if (verbose) {
            logln("--- Inputing vote vocab from " + file);
        }
        this.voteProperties = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line = reader.readLine();
        String[] sline = line.split("\t");
        for (int ii = 1; ii < sline.length; ii++) {
            this.voteProperties.add(sline[ii]);
        }

        this.voteVocab = new ArrayList<String>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            String vid = sline[0];
            Vote vote = new Vote(vid);
            for (int ii = 1; ii < sline.length; ii++) {
                vote.addProperty(voteProperties.get(ii - 1), sline[ii]);
            }
            this.voteVocab.add(vid);
            this.voteTable.put(vid, vote);
        }
        reader.close();
        if (verbose) {
            logln("--- --- # key votes: " + voteVocab.size());
        }
    }

    protected void outputAuthorVotes(String outputFolder) throws Exception {
        File authorVoteFile = new File(outputFolder, formatFilename + authorVoteExt);
        if (verbose) {
            logln("--- Outputing author votes to " + authorVoteFile);
        }
        BufferedWriter writer = IOUtils.getBufferedWriter(authorVoteFile);
        for (int aa = 0; aa < authorVocab.size(); aa++) {
            String author = authorVocab.get(aa);
            HashMap<Integer, Integer> aVotes = rawAuthorVotes.get(author);
            for (int vv : aVotes.keySet()) {
                writer.write(aa + "\t" + vv + "\t" + aVotes.get(vv) + "\n");
            }
        }
        writer.close();
    }

    protected void inputAuthorVotes(File file) throws Exception {
        if (verbose) {
            logln("--- Inputing author votes from " + file);
        }
        this.authorVotes = new HashMap[authorVocab.size()];
        for (int aa = 0; aa < this.authorVotes.length; aa++) {
            this.authorVotes[aa] = new HashMap<Integer, Integer>();
        }
        BufferedReader reader = IOUtils.getBufferedReader(file);
        String line;
        String[] sline;
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            int aid = Integer.parseInt(sline[0]);
            int vid = Integer.parseInt(sline[1]);
            int vote = Integer.parseInt(sline[2]);
            this.authorVotes[aid].put(vid, vote);
        }
        reader.close();

        this.votes = new int[authorVocab.size()][voteVocab.size()];
        for (int a = 0; a < authorVocab.size(); a++) {
            for (int b = 0; b < voteVocab.size(); b++) {
                Integer voteVal = this.authorVotes[a].get(b);
                if (voteVal == null) {
                    this.votes[a][b] = Vote.MISSING;
                } else {
                    this.votes[a][b] = voteVal;
                }
            }
        }
    }

    @Override
    protected void outputDocumentInfo(String outputFolder) throws Exception {
        // output document info
        File docInfoFile = new File(outputFolder, formatFilename + docInfoExt);
        if (verbose) {
            logln("--- Outputing document info ... " + docInfoFile);
        }
        BufferedWriter infoWriter = IOUtils.getBufferedWriter(docInfoFile);
        for (int docIndex : this.processedDocIndices) {
            String author = this.authorList.get(docIndex);
            int authorIdx = this.authorVocab.indexOf(author);
            if (authorIdx < 0) {
                throw new RuntimeException("Author " + author + " not found");
            }
            infoWriter.write(this.docIdList.get(docIndex)
                    + "\t" + authorIdx
                    + "\n");
        }
        infoWriter.close();
    }

    @Override
    public void inputDocumentInfo(File filepath) throws Exception {
        if (verbose) {
            logln("--- Reading document info from " + filepath);
        }

        // load authors and responses from doc info file
        BufferedReader reader = IOUtils.getBufferedReader(filepath);
        String line;
        String[] sline;
        ArrayList<String> dIdList = new ArrayList<String>();
        ArrayList<Integer> aList = new ArrayList<Integer>();

        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            dIdList.add(sline[0]);
            aList.add(Integer.parseInt(sline[1]));
        }
        reader.close();

        this.docIds = dIdList.toArray(new String[dIdList.size()]);
        this.authors = new int[aList.size()];
        for (int i = 0; i < this.authors.length; i++) {
            this.authors[i] = aList.get(i);
        }
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            inputAuthorVocab(new File(fFolder, formatFilename + speakerVocabExt));
            inputVoteVocab(new File(fFolder, formatFilename + voteVocabExt));
            inputAuthorVotes(new File(fFolder, formatFilename + authorVoteExt));

            File voteWordFile = new File(fFolder, formatFilename + voteTextExt);
            if (voteWordFile.exists()) {
                inputVoteTextData(voteWordFile);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    protected void inputAuthorVocab(File authorVocFile) throws Exception {
        if (verbose) {
            logln("Loading authors from vocab file " + authorVocFile);
        }

        this.authorProperties = new ArrayList<String>();
        BufferedReader reader = IOUtils.getBufferedReader(authorVocFile);
        String line = reader.readLine();
        String[] sline = line.split("\t");
        for (int ii = 1; ii < sline.length; ii++) {
            this.authorProperties.add(sline[ii]);
        }

        this.authorVocab = new ArrayList<String>();
        this.authorTable = new HashMap<String, Author>();
        while ((line = reader.readLine()) != null) {
            sline = line.split("\t");
            String id = sline[0];
            Author author = new Author(id);
            for (int ii = 1; ii < sline.length; ii++) {
                author.addProperty(authorProperties.get(ii - 1), sline[ii]);
            }
            this.authorVocab.add(id);
            this.authorTable.put(id, author);
        }
        reader.close();
    }
}
