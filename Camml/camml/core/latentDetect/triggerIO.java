package camml.core.latentDetect;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

public class triggerIO {

	/*
	 * This class is about the save all the triggers found into a local txt file.
	 * It means we store all triggers in advance 
	 * */
	
	private int varNum;
	
	public triggerIO(int varNum){
		
		this.varNum = varNum;
	}
	
	public void savetotxt(String filename, ArrayList<int[][]> matrices) throws IOException{
		
		BufferedWriter outputWriter = null;
		outputWriter = new BufferedWriter(new FileWriter(filename));  	
	 	
		for(int[][] currentMatrix : matrices)
		{
			for(int n = 0; n < varNum; n++)
			{
				for(int m = 0; m < varNum; m++)
				{
//					outputWriter.write(currentMatrix[n][m]);
					outputWriter.write(Integer.toString(currentMatrix[n][m]));
				}
				outputWriter.newLine();
			}
			outputWriter.write("***************");
			outputWriter.newLine();
		}
		
		outputWriter.flush();  
		outputWriter.close(); 
	}
	
	public ArrayList<int[][]> readtxt(String filename) throws IOException{
				
		BufferedReader br = new BufferedReader(new FileReader(filename));
		
		ArrayList<int[][]> matrices = new ArrayList<int[][]>();
		ArrayList<Integer> currentArray = new ArrayList<Integer>(); 
		String line = null;
		
//		line = br.readLine();
		while((line = br.readLine()) != null){
			
			if(line.length() <= varNum)
			{
				for(int i = 0; i < varNum; i++)
				{
					String word = String.valueOf(line.charAt(i)); 
					currentArray.add(Integer.parseInt(word));
				}
			}
			
 		    if(line.length() > varNum)
			{
				// read the one dimensional array into the two dimensional one:
			    // then add it to "matrices"
 		    	// 111 222 333 444 555 666 
				int[][] currentMatrix = new int[varNum][varNum];
				
				for(int i = 0; i < varNum; i++){
			
				    for(int n = 0; n < varNum; n++){
				    	currentMatrix[i][n] = currentArray.get(i*varNum + n);
				    }   
				}
				
				matrices.add(currentMatrix);				
				currentArray = new ArrayList<Integer>();
			}
		}
		br.close();
		return matrices;
	}

}
