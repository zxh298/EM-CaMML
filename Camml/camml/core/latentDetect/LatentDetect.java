/*
 *  [The "BSD license"]
 *  Copyright (c) Xuhui Zhang
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    3. The name of the author may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.*
 *
 *  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 *  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Package: camml.core.latentDetect
// Authors: Xuhui Zhang (xzha270@student.monash.edu)

package camml.core.latentDetect;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;
import java.util.TreeMap;

import camml.core.latentDetect.DataPreprocessing;
import cdms.core.Type;
import cdms.core.Value;
import cdms.core.VectorFN;

public class LatentDetect {

	/**
	 *  There should be at least FOUR variables in the dataset.
	 *  This is because the smallest trigger contains four variables.
	 *  
	 *  As five and six (maybe more) variable triggers all have four 
	 *  variable trigger structures as subnet, so we only need to check
	 *  any subnet of four variables match any four variable trigger.
	 * 
	 *  For example,  
	 *  
	 *  Five variable trigger:
	 *  
	 *   0 0 0 0 1 1
	 *   0 0 0 0 0 1
	 *   0 0 0 1 1 0
	 *   0 0 0 0 0 0 
	 *   0 0 0 0 0 0
	 *   0 0 0 0 0 0
	 *   
	 *  Has a subnet of a four variable trigger:
	 *  
	 *   0 0 0 1 1 
	 *   0 0 0 0 1
	 *   0 0 0 1 0
	 *   0 0 0 0 0
	 *   0 0 0 0 0
	 *   
	 *  For any variables left, CaMML will find how to connect them.
	 *  
	 * */
	
	private Value.Vector data;
	private double alpha;
	private int variableNum;
	private boolean triggerMatched;
	private final int subNetSize = 4;
	private ArrayList<int[]> subIndeces;
	// the index of the variables which conditional dependencies among them matched a trigger
	private int[] matchedSubIndex;
	// the rest variable index besides the matched ones
	private int[] remainedSubIndex;
	private int[][] hiddenModel;
	private double errorRate;
	private int[][] marginalDependencyMatrix;
	
	
	public LatentDetect(Value.Vector data, double alpha, double errorRate) throws IOException
	{
		this.data = data;
		this.alpha = alpha;
		this.variableNum = ((Value.Structured) data.elt(0)).length();
	    this.triggerMatched = false;
	    this.subIndeces = getSubVarList(subNetSize);
	    this.remainedSubIndex = new int[variableNum - subNetSize];
	    this.matchedSubIndex = null;
	    this.hiddenModel = null;
	    this.errorRate = errorRate;
	}

	public void run() throws IOException
	{
		for(int[] subIndex : subIndeces)
		{
//			double percent = (double)(subIndeces.indexOf(subIndex)+1)/subIndeces.size();
//			System.out.print(String.format("%.0f", percent*100));
			System.out.print("% ");
			
			Value.Vector sub_data = makePartialData(subIndex);
			DataPreprocessing dp = new DataPreprocessing(sub_data, errorRate, alpha, true);
			dp.run_analysis();
			triggerMatched = dp.getMatched();
			hiddenModel = dp.getHiddenModelForCurrentData();
			
			if(triggerMatched == true)
			{
				System.out.println();
//				System.out.println("Latent variable detected!");
				matchedSubIndex = subIndex;
				
				// get the remained index
				ArrayList<Integer> temp = new ArrayList<Integer>();
				for(int i=0; i<variableNum; i++)
				{
                    boolean in = false;
                    for(int n=0; n<matchedSubIndex.length; n++)
                    {
                    	if(matchedSubIndex[n] == i)
                    	{
                    		in = true;
                    	    break;
                    	}
                    }
                    if(in == true)
                    	continue;
                    else
                    	temp.add(i);
				}
				for(int i=0; i<temp.size(); i++)
				{
					remainedSubIndex[i] = temp.get(i);
				}
				
				break;
			} 
		}
		
		if(triggerMatched == false)
		{
			DataPreprocessing dp1 = new DataPreprocessing(data, errorRate, alpha, false);
			dp1.run_analysis();
			marginalDependencyMatrix = dp1.getCurrentDataDependencyMatrices().get(0);
		}
		
//		System.out.println();
	}
	
	public boolean getMatch()
	{
		return triggerMatched;
	}
	
	public int[] getMatchedSubsetIndeces()
	{
		return matchedSubIndex;
	}
	
	public int[] getRemainedSubsetIndeces() {
		return remainedSubIndex;
	}
	
	public int[][] getMatchedTriggerStructure()
	{
		return hiddenModel;
	}
	
	/**
	 * As all triggers have a subnet which is exactly the same as either one of the 
	 * 4 variable triggers, so we select 4 variables from the original dataset to do
	 * the detection. 
	 * 
	 * For any variables are being left, we use CaMML to decide how they will be
	 * connected. 
	 * 
	 * For example, if we choose 4 from 6 variable, there will be 6!/(4!*2!) = 15
	 * possible combinations.
	 * 
	 * */
	private ArrayList<int[]> getSubVarList(int subNum)
	{
		ArrayList<int[]> resultList = new ArrayList<int[]>();

		int[] iArr = new int[variableNum];

		for (int tempNum = 0; tempNum <= variableNum; tempNum++)
		{
			int num=0;
			int pos=0;
			int ppi=0;

			for(;;)
			{
				if(num==variableNum)
				{
					if(pos==1)    
						break;

					pos-=2;
					ppi=iArr[pos];
					num=ppi;
				}

				iArr[pos++]=++num;

				if(pos!=tempNum)
					continue;

				int[] tempArray = new int[pos];

				for(int i=0;i<pos;i++){

					tempArray[i] = iArr[i]-1;
				}
                	
				resultList.add(tempArray);
			}
		}

		// we only need the index with the given length 
		ArrayList<int[]> result = new ArrayList<int[]>();
		for(int[] index : resultList)
		{
			if(index.length == subNum)
				result.add(index);
		}
		
		return result;
	}
	
    // get subset of data by given a subset of variable indeces;
	private Value.Vector makePartialData(int[] partialVarIndex) {
		// get the headers of the original data:
		Type.Structured eltType = ((Type.Structured) (((Type.Vector) data.t).elt));
		String headers[] = eltType.labels;
		
		// number of selected variables:
        int partial_varNum = partialVarIndex.length;
		String headers_partial[] = new String[partialVarIndex.length];
		// add the selected headers in:
		for(int i=0; i<partialVarIndex.length; i++)
		{
			headers_partial[i] = headers[partialVarIndex[i]];
		}
		
		
		String[][] data_partial = new String[partialVarIndex.length][data.length()];
		TreeMap<String, Integer>[] columnValues = (TreeMap<String, Integer>[]) new TreeMap[partialVarIndex.length];

		for (int c = 0; c < partialVarIndex.length; c++) {
			columnValues[c] = new TreeMap<String, Integer>();
		}

		for (int n = 0; n < data.length(); n++) {
			for (int m = 0; m < partial_varNum; m++) {
				data_partial[m][n] = data.cmpnt(partialVarIndex[m]).elt(n).toString();
				columnValues[m].put(data_partial[m][n], 1);
			}
		}

		Value.Vector[] vecArray = new Value.Vector[headers_partial.length];

		for (int c = 0; c < headers_partial.length; c++) {
			Set<String> keysPre = columnValues[c].keySet();
			String[] keys = new String[keysPre.size()];
			keysPre.toArray(keys);

			for (int i = 0; i < keys.length; i++) {
				columnValues[c].put(keys[i], i);
			}

			Type.Symbolic type = new Type.Symbolic(false, false, false, false, keys);

			int[] intArray = new int[data_partial[c].length];
			for (int j = 0; j < intArray.length; j++) {
				String v = data_partial[c][j];
				intArray[j] = columnValues[c].get(v);
			}

			// combine type and value together to form a vector of symbolic values.
			vecArray[c] = new VectorFN.FastDiscreteVector(intArray, type);
		}

		Value.Structured vecStruct = new Value.DefStructured(vecArray, headers_partial);
		Value.Vector vec = new VectorFN.MultiCol(vecStruct);

		return vec;
	}
	
	public boolean triggerMatched()
	{
		return triggerMatched;
	}
	
	// This is for initializing start structure of EM-CaMML
	public int[][] getMarginalDependencyMatrix(){
		return marginalDependencyMatrix;
	}
	
//	public int[][] getMarginalDependencyMatrix() {
//		
//	}
//	
}
