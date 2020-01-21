package camml.core.latentDetect;

import java.util.ArrayList;

/**
 *  This class is used to check all collisions in a 
 *  given DAG
 *  
 *  For example:
 *  
 *  A->B<-C->D<-E
 * 
 *  B and D are collisions in this DAG
 *  
 * */


public class GetCollisions {

	private static int[][] DAG;

	private static int varNum;

	public GetCollisions(int[][] currentDAG, int currentVarNum)
	{
		varNum = currentVarNum;
		DAG = new int[varNum][varNum];
		// Initialize the DAG:
		for(int n = 0; n < varNum; n++)
		{
			for(int m = 0; m < varNum; m++)
			{
				DAG[n][m] = currentDAG[n][m];
			}
		}
	}

	/**
	 *  For example:
	 * 
	 *  For the DAG matrix:   
	 *  
	 *  0 1 0 0  
	 *  0 0 1 1 
	 *  0 0 0 1
	 *  0 0 0 0
	 *  
	 *  The fourth variable is the common effect (collision) 
	 *  of the second and third variables.
	 *  
	 *  In a DAG, there may be more than one collisions, so
	 *  the format of result should be a list 
	 * */

	public static ArrayList<Integer> returnCollisions()
	{
		ArrayList results = new ArrayList<Integer>();

		for(int m = 0; m < varNum; m++)
		{
			// for each variable (column) chech the number of 
			// arcs. If the number is bigger than 1, than the 
			// variable is a collision
			int arcNum = 0;

			for(int n = 0; n < varNum; n++)
			{
				if (DAG[n][m] == 1)
					arcNum++;
			}

			if (arcNum > 1)
				// add the current variable as a collision
				results.add(m+1);
		}

		return results;
	}

	// Get all children of the given collision:
	public static ArrayList<Integer> getCurrentCollisionDescendants(int currentCollision)
	{
		ArrayList<Integer> results = new ArrayList<Integer>();

		// get all the parent nodes 
		ArrayList<Integer> parentNodes = getCurrentCollisionParents(currentCollision);

		// store all non-parent nodes for the current collision,
		// then search the descendants from them:
		ArrayList<Integer> nonParentNodes = new ArrayList<Integer>();				
		for(int n = 1; n <= varNum; n++)
		{
			if(n == currentCollision)
				continue;

			boolean flag = true;

			for(Integer parent : parentNodes)
			{
				// if the current variable is the collision's parent,
				// then set false to false:
				if(parent == n)
					flag = false;
			}

			if(flag == true)
				nonParentNodes.add(n);		
		}

		for(Integer nonParentNode : nonParentNodes)
		{
			ArrayList<ArrayList<Integer>> paths = new ArrayList<ArrayList<Integer>>();
			//System.out.println("new paths size:" + paths.size());
			paths = new GetAllPaths(DAG, varNum, currentCollision, nonParentNode, true).getResult();

			if(paths.size() > 0)
				results.add(nonParentNode);

		}

		return results;
	}

	// Get all parents of the given collision:
	public static ArrayList<Integer> getCurrentCollisionParents(int currentCollision)
	{
		ArrayList<Integer> results = new ArrayList<Integer>();

		for(int n = 0; n < varNum; n++)
		{
			if (DAG[n][currentCollision-1] == 1)
				results.add(n+1);
		}

		return results;
	}

	public static void main(String[] args)
	{
		int[][] DAG = new int[5][5];

		for(int n = 0; n < 5; n++)
		{
			for(int m = 0; m < 5; m++)
			{
				DAG[n][m] = 0;
			}
		}

		//DAG[0][1] = 1;
		DAG[0][2] = 1;
		DAG[1][2] = 1;
		DAG[1][3] = 1;
		DAG[2][4] = 1;
		//DAG[2][6] = 1;
		//DAG[5][6] = 1;
		//DAG[4][7] = 1;
		//DAG[2][3] = 1;

		ArrayList<Integer> results = new GetCollisions(DAG,5).returnCollisions();

		for(Integer result: results)
		{
			System.out.println(result);
		}

		System.out.println("*********************");

		ArrayList<Integer> descendants = new GetCollisions(DAG,5).getCurrentCollisionDescendants(3);

		ArrayList<Integer> parents = new GetCollisions(DAG,5).getCurrentCollisionParents(3);

		System.out.println("Children:");

		for(Integer descendant: descendants)
		{
			System.out.println(descendant);
		}

		System.out.println("Parents:");

		for(Integer parent: parents)
		{
			System.out.println(parent);
		}
	}

}
