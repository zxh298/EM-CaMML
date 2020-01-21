package camml.core.latentDetect;

import java.util.ArrayList;

public class GenerateAllDependencies {

	int[][] currentDAG;

	int varNum;
	// to specify whether the current DAG contains a hidden variable (the first variable in the DAG)
	boolean containHiddenVar;

	public GenerateAllDependencies(int[][]DAG, int varNum, boolean containHiddenVar)
	{
		this.varNum = varNum;

		currentDAG = new int[varNum][varNum];

		for(int n = 0; n < varNum; n++)
		{
			for(int m = 0; m < varNum; m++)
			{
				currentDAG[n][m] = DAG[n][m];
			}
		}

		this.containHiddenVar = containHiddenVar;
	}

	private ArrayList<int[]> getConditionVarList()
	{

		/**
		 * Generate all possible condition variable set (by setting currentVarNum to varNum):
		 * 
		 * For example:
		 * 
		 * For variable A, B, C in a given DAG,
		 * the condition variable set are:
		 * Ø(0), 1, 2, 3, 12, 13, 23, 123
		 * 
		 * If the current DAG contains a hidden variable (the first variable), then in this case,
		 * the condition variable set are:
		 * Ø(0), 2, 3, 4, 23, 24, 24, 234
		 *  
		 * */
		int newVarNum = varNum;
		// because we dont condition on the hidden variable, thus varNum should 
		// change to varNum - 1:
		if(containHiddenVar == true)
		{
			newVarNum = newVarNum - 1;
		}

		ArrayList<int[]> resultList = new ArrayList<int[]>();

		int[] iArr = new int[newVarNum];

		// Set the empty variable set Ø by 0
		int[] fisrtEmptySet = new int[1];
		fisrtEmptySet[0] = 0;
		resultList.add(fisrtEmptySet);

		for (int tempNum = 1; tempNum < newVarNum+1; tempNum++)
		{
			int num=0;
			int pos=0;
			int ppi=0;

			for(;;)
			{
				if(num==newVarNum)
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

					tempArray[i] = iArr[i];
				}

				resultList.add(tempArray);

				//System.out.println();
			}
		}

		if(containHiddenVar == true)
		{
			int num = 0;
			for(int[] currentConditionSet : resultList)
			{
				// because the first one is Ø(0) which should not be changed
				if(num > 0)
				{
					//int[] temp = new int[currentConditionSet.length];
					for(int i = 0; i < currentConditionSet.length; i++)
					{					
						currentConditionSet[i] = currentConditionSet[i] + 1;
					}					
				}
				num++;
			}			
		}

		return resultList;
	}

	private ArrayList<int[]> getRemainingVarList()
	{
		/**
		 *  Generate the remaining variable list which will be 
		 *  applied D-separation rules.
		 *  
		 *  For example:
		 *  
		 *  for a 4 variable matrix
		 *  
		 *    condition         remaining
		 *        Ø(0)          1, 2, 3, 4          
		 *        1             2, 3, 4
		 *        2             1, 3, 4
		 *        3             1, 2, 4 
		 *        4             2, 3, 4
		 *        1, 2          3, 4
		 *        1, 3          2, 4
		 *        1, 4          2, 3
		 *        2, 3          1, 4
		 *        2, 4          1, 3
		 *        3, 4          1, 2
		 *     1, 2, 3          4
		 *     1, 2, 4          3
		 *     1, 3, 4          2
		 *     2, 3, 4          1  
		 *    1, 2, 3, 4        Ø(0)       
		 * */

		ArrayList<int[]> resultList = new ArrayList<int[]>();

		ArrayList<int[]> conditionOnVarList = getConditionVarList();

		// The first remaining variable list should be the full list:		
		int[] firstResult = new int[varNum];        
		for(int i = 0; i < varNum; i++)
		{
			firstResult[i] = i+1;
		}       
		resultList.add(firstResult);

		int num = 0;

		for(int[] currentConditionOnVarList : conditionOnVarList)
		{

			// cuz the first list from resultList is null(0), so skip it.
			if(num == 0)
			{
				num++;
				continue;
			}

			ArrayList<Integer> fullVarArrayList = new ArrayList<Integer>();

			for(int i = 1; i <= varNum; i++)
			{
				fullVarArrayList.add(i);
			}

			for(int var : currentConditionOnVarList)
			{	
				for(int i = 0; i < fullVarArrayList.size(); i++)
				{
					// remove the variables from fullVarArrayList because they exist 
					// in currentConditionOnVarList
					if(fullVarArrayList.get(i) == var)
						fullVarArrayList.remove(i);
				}
			}

			int[] tempResult = new int[fullVarArrayList.size()];
			// transfer the ArrayList to int[]:
			for(int i = 0; i < tempResult.length; i++)
			{
				tempResult[i] = fullVarArrayList.get(i);
			}

			if(tempResult.length == 0)
			{
				int[] finalResult = new int[1];
				finalResult[0] = 0;
				resultList.add(finalResult);
			}
			else 
				resultList.add(tempResult);

		}

		return resultList;
	}

	private ArrayList<int[]> getTwoVarDSeparateCombinationList(int[] currentRemainList)
	{
		/**
		 *  Get all combination of two variables that will be applied 
		 *  D-Separation rules to check their dependencies. 
		 *  
		 *  For example:
		 *  If the current remaining variables are 1, 2, 3
		 *  So all the combinations are: 
		 *  
		 *  1,2    1,3     2,3
		 * */

		ArrayList<int[]> result = new ArrayList<int[]>();

		for(int n = 0; n < currentRemainList.length; n++)
		{			
			for(int m = n+1; m < currentRemainList.length; m++)
			{
				int[] temp = new int[2]; 
				temp[0] = currentRemainList[n];
				temp[1] = currentRemainList[m];

				int num1 = currentRemainList[n];
				int num2 = currentRemainList[m];

				result.add(temp);
			}						
		}

		return result;
	}

	private boolean collisionBlocksThePath(int currentCollision, int[] currentConditionOnVars, ArrayList<Integer> currentPath)
	{
		/**
		 *  0: independent    1: dependent
		 *  
		 *  Check whether there exist collisions in the current path:
		 * 
		 *  1) For the current collision, the nodes before it and after it on the current path
		 *     should be its parents, or the current collision should not be treat as a collision
		 *     of the current path. 
		 *     
		 *     For example:
		 *     If the current path from A to E is A-B-C-D-E, and C is a collision of the current DAG,
		 *     but at least one of B and D is not a parent of C, so C cannot be treat as a collision 
		 *     on this path.
		 *     
		 *  2) If the collisions or their descendants are contained by the current   
		 *     condition variables, the current path is not being blocked (dependent).  
		 *     
		 *  3) If both collisions and their descendants are not contained from the     
		 *     current condition variables, the current path is being blocked.
		 * */

		boolean block = true;

		ArrayList<Integer> currentCollisionParents = new GetCollisions(currentDAG,varNum).getCurrentCollisionParents(currentCollision);
		ArrayList<Integer> currentCollisionDescendants = new GetCollisions(currentDAG,varNum).getCurrentCollisionDescendants(currentCollision);

		int beforeCollisionNodeOnPath = currentPath.get(currentPath.indexOf(currentCollision)-1); 
		int afterCollisionNodeOnPath = currentPath.get(currentPath.indexOf(currentCollision)+1);

		if(currentCollisionParents.contains(beforeCollisionNodeOnPath) && currentCollisionParents.contains(afterCollisionNodeOnPath))
		{
			for(int currentConditionVar : currentConditionOnVars)
			{
				// if the current collision itself is on condition, then block is false: 
				if(currentCollision == currentConditionVar)
				{
					block = false;
					break;
				}

				// if any of the descendant of current collision is on condition, then block is false:
				for(int currentDescendant : currentCollisionDescendants)
				{
					if(currentDescendant == currentConditionVar)
					{
						block = false;
						break;
					}
				}
			}
		}

		else
		{
			// treat the current collision as a normal node (a node in a chain)
			for(int currentConditionVar : currentConditionOnVars)
			{
				if(currentCollision == currentConditionVar)
				{
					block = true;
					break;
				}
				else
				{
					block = false;
				}
			}
		}


		return block;
	}

	public ArrayList<int[][]> getResult()
	{
		/**
		 *  Generate all dependencies between every two variables
		 *  by applying D-separation rules:
		 *  1. chain   2. common cause   3. common effect 
		 *    
		 *  For example, in the following matrix (0:independent 1:dependent):
		 *  
		 *    1 2 3 4
		 * ---------
		 * 1| 0 1 0 0 
		 * 2| 0 0 1 0
		 * 3| 0 0 0 0
		 * 4| 0 0 0 0
		 *  
		 * 1 and 2, 2 and 3 are dependent on each other  
		 *  
		 * One DAG has many condition variable set with its corresponding remaining variable set;
		 * One condition variable set has one dependency matrix;
		 * So each DAG has many dependency matrices. 
		 * 
		 * */

		// the result is a set of two dimensional dependency matrix for a single current DAG
		ArrayList<int[][]> resultDependenciesForCurrentDAG = new ArrayList<int[][]>();
		// get the variable list that being conditioned on: 
		ArrayList<int[]> conditionOnVarList = getConditionVarList();
		// get the remaining variables except the ones from conditionalVarList:
		ArrayList<int[]> remainVarList = getRemainingVarList();
		// get the collisions of the current DAG
		ArrayList<Integer> allCollisions = new GetCollisions(currentDAG,varNum).returnCollisions();

		for(int i = 0; i < remainVarList.size(); i++)
		{
			// get every set of two remaining variables from remainVarList
			int[] currentRemainVariables = remainVarList.get(i);

			// the variables number of the current remaining set should be bigger than one,
			// if the current model contains a hidden variable, then should be bigger than 
			// three otherwise the remaining variables' number will be bigger than the corresponding
			// observed model.
			if(containHiddenVar == false)
			{

				if(currentRemainVariables.length < 2)
					continue;
				//				else
				//				{
				//					for(int t = 0; t < currentRemainVariables.length; t++)
				//						System.out.print(currentRemainVariables[t]+" ");
				//					System.out.println();
				//				}
			}

			if(containHiddenVar == true)
			{
				if(currentRemainVariables.length < 3)
					continue;
				//				else
				//				{
				//					for(int t = 0; t < currentRemainVariables.length; t++)
				//						System.out.print(currentRemainVariables[t]+" ");
				//					System.out.println();
				//				}
			}

			// get every two variables combinations list from the ith set of remained variables, and which will
			// be checked the dependencies by applying D-Separation rules:      	
			ArrayList<int[]> currentTwoVarCombinationsList = getTwoVarDSeparateCombinationList(currentRemainVariables);

			// get the corresponding ith condition variable set from conditionOn
			int[] currentConditionVariables = conditionOnVarList.get(i);


			// the current result dependency matrix for the current remained variables
			int[][] currentDependencyMatrix = new int[varNum][varNum];
			for(int n = 0; n < varNum; n++)
			{
				for(int m = 0; m < varNum; m++)
				{
					currentDependencyMatrix[n][m] = 0;
				}        		
			}


			// for every set of two variables from the current combination list:
			for(int[] currentTwoVarCombination : currentTwoVarCombinationsList)
			{
				// get every possible path between the current two variables (false: ignoring the arc directions)
				GetAllPaths gp = new GetAllPaths(currentDAG, varNum, currentTwoVarCombination[0], currentTwoVarCombination[1], false);
				ArrayList<ArrayList<Integer>> paths = gp.getResult();

				// check if there is at least one path is not being blocked by current condition variables
				// if there exists at least one path noting being blocked, these two current remaining variables
				// are dependent.
				int blockedNum = 0;

				// for every path:
				for(ArrayList<Integer> path : paths)
				{										
					// to show whether the path is being blocked 
					//boolean beingBlocked = false;
					boolean beingBlockedByCollision = false;
					boolean beingBlockedByNonCollision = false;
					/**
					 * If the path only contains the two remaining variables, set them dependent
					 * 
					 * For example:
					 * if the two remaining variables are A and B, and we try to check whether they 
					 * are dependent (at least there is one path between them is not being blocked) 
					 * or not, and one of the path only contains A and B (A->B), then we set them 
					 * dependent directly.
					 *
					 * */										
					if(path.size() == 2)
						break;

					/**
					 *  If the path's size is bigger than 2 (there are other variables beside the two 
					 *  remaining variables), remove the two remaining variables (the first and last 
					 *  variable) from the path, because we dont need to condition on them.
					 *  
					 * */
					ArrayList<Integer> newPath = (ArrayList<Integer>) path.clone();
					newPath.remove(0);
					newPath.remove(newPath.size()-1);


					//	for(int conditionVar : currentConditionVariables)
					//	{
					//		if (path.contains(conditionVar))						
					//	   {
					//		 // if any of the condition variable is contained in the current path, it means
					//	     // that the current path is blocked by the current condition variable set
					//		 blockedNum++;
					//		 break;
					//		}						
					//	}

					for (int currentNode: newPath)
					{
						// if the current node is a collision:
						if(allCollisions.contains(currentNode))
						{
							// the third parameter should be path instead of newPath because the parents of the current
							// collisions could contain the first(START) node or last(END) node of path.
							beingBlockedByCollision = collisionBlocksThePath(currentNode, currentConditionVariables, path);	

							if(beingBlockedByCollision == true)
							{
								blockedNum++;
								break;
							}
						}

						// if the current node is not a collision:
						else
						{
							for(int currentConditionVar : currentConditionVariables)
							{
								if(currentNode == currentConditionVar)
								{
									beingBlockedByNonCollision = true;
									break;
								}
							}

							if (beingBlockedByNonCollision == true)
							{
								blockedNum++;
								break;
							}
						}
					}


				}
				// if the number of blocked paths is smaller than the total number of paths, it means there exists 
				// a certain number of paths are not being blocked by the current condition variables and the two
				// remaining variables are dependent
				if(blockedNum < paths.size())
				{
					// the two variables are dependent on each other(by setting to 1):
					currentDependencyMatrix[currentTwoVarCombination[0]- 1][currentTwoVarCombination[1]- 1] = 1;
					currentDependencyMatrix[currentTwoVarCombination[1]- 1][currentTwoVarCombination[0]- 1] = 1;
				}								
			}

			resultDependenciesForCurrentDAG.add(currentDependencyMatrix);
		}

		// GetAllPaths(int[][] currentMatrix, int currentVarNum, int startVar, int endVar, boolean flag)

		return resultDependenciesForCurrentDAG;		
	}

	public static void main(String[] args)
	{
		/**
		 * If you want test DAG with hidden variable:
		 * 
		 * 1: change int[][] DAG = new int[3][3] to 
		 *           int[][] DAG = new int[4][4]
		 *           
		 * 2: change GenerateAllDependencies(DAG,3,false) to
		 *           GenerateAllDependencies(DAG,4,true)
		 * */
		System.out.println("#############################");
		System.out.println("0: Independent   1: Dependent");
		System.out.println("#############################");
		System.out.println();

//		DAGList dl = new DAGList(3);
//
//		dl.makeMatrixList();
//
//		//Generate all possible DAGs for the current variable number:
//		ArrayList<int[][]> observedDAGList = dl.getResultList();

		
      
		//*************TEST OBSERVED DAG*************
//		int[][] observedDAG = new int[5][5];
//		for(int n = 0; n < 5; n++)
//		{
//			for(int m = 0; m < 5; m++)
//			{
//				observedDAG[n][m] = 0;
//			}
//		}
//		observedDAG[0][2] = 1;
//		observedDAG[1][2] = 1;
//		observedDAG[2][3] = 1;
//		observedDAG[2][4] = 1;
//		
//		System.out.println("Current observed DAG:");
//		for(int n = 0; n < observedDAG.length; n++)
//		{
//			for(int m = 0; m < observedDAG.length; m++)
//			{
//				System.out.print(observedDAG[n][m] + " ");
//			}
//			System.out.println();
//		}
//		System.out.println("-----------------------");
//		
//		GenerateAllDependencies gd = new GenerateAllDependencies(observedDAG,5,false);
//		ArrayList<int[]> conditionResults = gd.getConditionVarList();
//
//		ArrayList<int[][]> finalResult = gd.getResult();
//		System.out.println("Size: " + finalResult.size());
//		
//		for(int i = 0; i < finalResult.size(); i++)
//		{
//			System.out.print("Condition Variables: ");
//			int[] conditionResult = conditionResults.get(i);
//			// For both model contains hidden variable or not:
//			// if there are three variables, set conditionResult.length < 2;
//			// if there are four variables, set conditionResult.length < 3
//			// if there are five variables, set conditionResult.length < 4
//			// if there are six variables, set conditionResult.length < 5
//			// because the number of remaining variables should at least 
//			// be more than 2 so that can be tested whether they are dependent 
//			// or not
//
//			if(conditionResult.length < 4)
//			{
//				for (int n = 0 ; n < conditionResult.length; n++)
//				{
//					// if the current model contains hidden variable, then the result 
//					// condition variable is real variable number plus one
//
//					System.out.print(conditionResult[n] + " ");
//				}
//
//				System.out.println();
//			}
//
//			int[][] currentMatrix = finalResult.get(i);
//			for(int n = 0; n < 5; n++)
//			{
//				for(int m = 0; m < 5; m++)
//					System.out.print(currentMatrix[n][m] + " ");
//				System.out.println();
//			}
//
//			System.out.println("-----------------------");
//		}
		
        //*******************************************
		
		System.out.println("*************************");
		
		//*********TEST HIDDEN MODEL*****************
		int[][] hiddenDAG = new int[5][5];
		for(int n = 0; n < 5; n++)
		{
			for(int m = 0; m < 5; m++)
			{
				hiddenDAG[n][m] = 0;
			}
		}
		
		// the hidden variable is connected to 2nd and 5th variables
		hiddenDAG[0][3] = 1;
		hiddenDAG[0][4] = 1;
		hiddenDAG[1][2] = 1;
		hiddenDAG[1][4] = 1;
		hiddenDAG[2][3] = 1;
		
		System.out.println("Current hidden DAG:");
		
		for(int n = 0; n < hiddenDAG.length; n++)
		{
			for(int m = 0; m < hiddenDAG.length; m++)
			{
				System.out.print(hiddenDAG[n][m] + " ");
			}
			System.out.println();
		}
		System.out.println("-----------------------");
		
		GenerateAllDependencies gd1 = new GenerateAllDependencies(hiddenDAG,5,true);
		ArrayList<int[]> conditionResults1 = gd1.getConditionVarList();

		ArrayList<int[][]> finalResult1 = gd1.getResult();
		System.out.println("Size: " + finalResult1.size());
		
		for(int i = 0; i < finalResult1.size(); i++)
		{
			System.out.print("Condition Variables: ");
			int[] conditionResult1 = conditionResults1.get(i);
			// For both model contains hidden variable or not:
			// if there are three variables, set conditionResult.length < 2;
			// if there are four variables, set conditionResult.length < 3
			// if there are five variables, set conditionResult.length < 4
			// if there are six variables, set conditionResult.length < 5
			// because the number of remaining variables should at least 
			// be more than 2 so that can be tested whether they are dependent 
			// or not

			if(conditionResult1.length < 4)
			{
				for (int n = 0 ; n < conditionResult1.length; n++)
				{
					// if the current model contains hidden variable, then the result 
					// condition variable is real variable number plus one

					System.out.print(conditionResult1[n] + " ");
				}

				System.out.println();
			}

			int[][] currentMatrix1 = finalResult1.get(i);
			for(int n = 0; n < 5; n++)
			{
				for(int m = 0; m < 5; m++)
					System.out.print(currentMatrix1[n][m] + " ");
				System.out.println();
			}

			System.out.println("-----------------------");
		}
		//*******************************************

	}
}
