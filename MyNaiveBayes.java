package CS286;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.StringTokenizer;


public class MyNaiveBayes {
	
	static HashMap<String,Integer> spamMap = new HashMap<String,Integer>();
	static HashMap<String,Integer> hamMap = new HashMap<String,Integer>();
	static HashSet<String> totalWords = new HashSet<String>();
	static float pSpam = 0.0f;
	static float pHam = 0.0f;
	static int count =0;
	static Writer output;
	static String outText="";
	
	public static void main(String args[]) throws FileNotFoundException, IOException{
	
	
		
		String type = args[0];
		String model_dir = "", ham_dir="", spam_dir="", input_file="";
		if(type.equals("build")){
			ham_dir = args[1];
			spam_dir = args[2];
			model_dir = args[3];
		} else{
			model_dir = args[1];
			input_file = args[2];
		}
		
		
		//To DO
		//String type ="predict";
		
		if(type.equals("build")){
			
			output = new BufferedWriter(new FileWriter(model_dir, true));
			//output = new BufferedWriter(new FileWriter("test.txt", true));
			ArrayList<String> hamSubjects = new ArrayList<String>();
			ArrayList<String> spamSubjects = new ArrayList<String>();
			File hamDirectory = new File(ham_dir);
			for (final File fileEntry : hamDirectory.listFiles()) {
				try (BufferedReader br = new BufferedReader(new FileReader(fileEntry))) {
					String line;
					while ((line = br.readLine()) != null) {
						if(line.contains("Subject:")){
							hamSubjects.add("ham||"+line.substring(8).trim());
						}
					}
				}
		    }
			File spamDirectory = new File(spam_dir);
			for (final File fileEntry : spamDirectory.listFiles()) {
				try (BufferedReader br = new BufferedReader(new FileReader(fileEntry))) {
					String line;
					while ((line = br.readLine()) != null) {
						if(line.contains("Subject:")){
							spamSubjects.add("spam||"+line.substring(8).trim());
						}
					}
				}
		    }
			
			// to make sure that spam and ham data are spread across in the list
			ArrayList<String> subjects = new ArrayList<String>();
			subjects.addAll(spamSubjects);
			subjects.addAll(hamSubjects);
			Collections.shuffle(subjects);
			// divide the subjects lists into a training and testing data : 70%,30%
			List<String> trainingData = partition(true,subjects,100);
			// Now train the model
			for(int i=0;i<trainingData.size();i++){
				String sub = trainingData.get(i);
				StringTokenizer st = new StringTokenizer(sub,"||");
				String temp[] = new String[2];
				temp[0] = st.nextToken();
				temp[1] = st.nextToken().replace("\\s+"," ").trim();
				if(temp[0].equals("spam")){
					String str[] = removeNoise(temp[1]).split(" ");
					for(String p :str){
						if(spamMap.containsKey(p.trim())){
							spamMap.put(p.trim(), spamMap.get(p.trim())+1);
						}
						else{
							spamMap.put(p.trim(),1);
						}
					}
				}else{
					String str[] = removeNoise(temp[1]).split(" ");
					for(String p :str){
						if(hamMap.containsKey(p.trim())){
							hamMap.put(p.trim(), hamMap.get(p.trim())+1);
						}
						else{
							hamMap.put(p.trim(),1);
						}
					}
				}
			}
			// find total no of unique words
			for(int i=0;i<subjects.size();i++){
				StringTokenizer st = new StringTokenizer(subjects.get(i),"||");
				String temp[] = new String[2];
				temp[0] = st.nextToken();
				temp[1] = st.nextToken().replace("\\s+"," ").trim();
				String str[] = removeNoise(temp[1]).split(" ");
				for(String p :str){
					totalWords.add(p);
				}
			}
			
//			System.out.println("Training Data size:"+ trainingData.size());
//			System.out.println("testing Data size:"+ testingData.size());
//			System.out.println("Vocabulary :"+ totalWords.size());
			
			// calculate pHam
			pHam = (float)hamMap.size()/(float)(hamMap.size()+spamMap.size());
			//System.out.println(pHam);
			//calculate pSpam
			pSpam = (float)spamMap.size()/(float)(hamMap.size()+spamMap.size());
			//System.out.println(pSpam);
			outText = outText + pHam+"\n";
			outText = outText + pSpam+"\n";
			outText = outText + spamMap.size()+"\n";
			outText = outText + hamMap.size()+"\n";
			outText = outText + totalWords.size()+"\n";
			train(trainingData);
			//System.out.println(outText);
			FileWriter fw = new FileWriter(model_dir);
			//FileWriter fw = new FileWriter("test.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(outText);
			bw.close();
            
		}
		
		else if(type.equals("predict")){
			HashMap<String,Word> hm = new HashMap<String,Word>();
			float pSpam = 1.0f, pHam = 1.0f;
			int spamSize =0;
			int hamSize = 0;
			int vocabulary =0;
			try (BufferedReader br = new BufferedReader(new FileReader(model_dir))) {
			//try (BufferedReader br = new BufferedReader(new FileReader("test.txt"))) {
				String line;
				int count = 0;
				pHam = Float.parseFloat(br.readLine());
				pSpam = Float.parseFloat(br.readLine());
				spamSize = Integer.parseInt(br.readLine());
				hamSize = Integer.parseInt(br.readLine());
				vocabulary = Integer.parseInt(br.readLine());
			
				while ((line = br.readLine()) != null) {
					if(line!=""){
						StringTokenizer st = new StringTokenizer(line,";");
						//System.out.println(line);
						Word w = new Word();
						String word = st.nextToken();
						w.setWord(word);
						w.setSpamFreq(Integer.parseInt(st.nextToken()));
						w.setHamFreq(Integer.parseInt(st.nextToken()));
						hm.put(word,w);
					}
				}
			}
			try (BufferedReader br = new BufferedReader(new FileReader(input_file))) {
				String line;
				while ((line = br.readLine()) != null) {
					if(line.contains("Subject:")){
						line = removeNoise(line.substring(8).replace("\\s+"," ").trim()).trim();
						break;
					}
				}
				float spam = 1.0f, ham =1.0f;
				String tokens[] = line.split(" ");
				for(String token : tokens){
					if(!token.equals("")){
						Word w = hm.get(token);
						spam = spam*((float)(1+w.getSpamFreq())/(float)(spamSize+vocabulary));
						ham = ham*((float)(1+w.getHamFreq())/(float)(hamSize+vocabulary));
					}
				}
				
				spam = pSpam * spam;
				ham = pHam * ham;
				
				if(spam>ham){
					System.out.println("classify=spam");
				} else{
					System.out.println("classify=ham");
				}
			}
			
		}
	}
	
	static void train(List<String> trainingData) throws IOException{
		
				int truePositive = 0, trueNegative = 0, falsePositive =0, falseNegative =0;
				for(int i=0;i<trainingData.size();i++) {
					String sub = trainingData.get(i);
					//System.out.println("-->"+ testingData.get(i));
					StringTokenizer st = new StringTokenizer(trainingData.get(i),"||");
					String temp[] = new String[2];
					temp[0] = st.nextToken();
					temp[1] = st.nextToken().replace("\\s+"," ").trim();
					
					boolean spam = check(temp[1],true);
					//System.out.println(spam);
					if("spam".equals(temp[0])){
						if(spam){
							truePositive++;
						}else{
							falsePositive++;
						}
					}
					if("ham".equals(temp[0])){
						if(spam){
							falseNegative++;
						}else{
							trueNegative++;
						}
					}
				}
				float accuracy = (float)(truePositive+trueNegative)/(float)(truePositive+trueNegative+falseNegative+falsePositive);
				System.out.println("Accuracy:"+accuracy);
	}
	

	
	static List<String> partition(boolean flag,ArrayList<String> subjects, int percentage){
		if(flag){
			return subjects.subList(0, (subjects.size()*percentage)/100);
		}
		else{
			return subjects.subList((subjects.size()*percentage)/100,subjects.size());
		}
	}
	
	static boolean check(String words, boolean build) throws IOException{
		float spam = 1.0f, ham =1.0f;
		String[] tokens = removeNoise(words.trim().replace("\\s+"," ")).split(" ");
		for(String word : tokens){
			int sTermFreq = (spamMap.get(word.trim())!=null) ? spamMap.get(word.trim()) : 0;
			int hTermFreq = (hamMap.get(word.trim())!=null) ? hamMap.get(word.trim()) : 0;
			spam = spam * (float)((1+sTermFreq)/(float)(spamMap.size()+totalWords.size()));
			ham = ham * (float)((1+hTermFreq)/(float)(hamMap.size()+totalWords.size()));
			if(build && !word.equals("")){
				count++;
				outText = outText+word.trim()+";"+sTermFreq+";"+hTermFreq+"\n";
			}
		}
		spam = pSpam * spam;
		ham = pHam * ham;
		return spam > ham;
	}
	
	static String removeNoise(String s) {
		String [] str = { ",", "#", ";", "\"", "\'", "!", ".", "[", "]", "(" , ")" , "+", "--", "?", ":", "/", "-", "*"};
		for (String p : str) {
			if (s.contains(p)) {
				s = s.replace(p,"");
			}
		}
		return s.toLowerCase();
	}
	

}

class Word {

	String word;
	int spamFreq;
	int hamFreq;
	public String getWord() {
		return word;
	}
	public void setWord(String word) {
		this.word = word;
	}
	public int getSpamFreq() {
		return spamFreq;
	}
	public void setSpamFreq(int spamFreq) {
		this.spamFreq = spamFreq;
	}
	public int getHamFreq() {
		return hamFreq;
	}
	public void setHamFreq(int hamFreq) {
		this.hamFreq = hamFreq;
	}
	
	
	
}




