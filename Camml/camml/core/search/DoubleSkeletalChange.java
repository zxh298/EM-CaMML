/*
 *  [The "BSD license"]
 *  Copyright (c) 2002-2011, Rodney O'Donnell, Lucas Hope, Lloyd Allison, Kevin Korb
 *  Copyright (c) 2002-2011, Monash University
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

//
// Double Skeletal Change operation for mutating TOMs
//

// File: DoubleSkeletalChange
// Author: {rodo,lhope}@csse.monash.edu.au

package camml.core.search;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;


/**
 * DoubleSkeletalChange chooses 3 variables at random such that a -> b -> c in the total ordering.
 * The arcs a->c and b->c are then toggled.  This can either add/remove two arcs or perform a
 * parent swap. 
 */
public class DoubleSkeletalChange extends TOMTransformation {
    
    /** cost to add an arc. */
    protected double arcCost;
    
    /** pre-allocated return value for getNodesChanged() */
    private int[] changes = new int[1];
    
    /** Pre-allocated value to pass to TOMCoster.toggleArcs()*/
    private int[][] nodesToggled = new int[2][2];
    
    /** Return array of changes nodes <br> 
     *  NOTE: Values within returned array are volatile and will often be changed is transform()
     *        is called again.  */
    public int[] getNodesChanged() { return changes; }
    
    /**
     * Constructor for DoubleSkeletalChange.
     */
    public DoubleSkeletalChange(Random generator, double arcProb, 
                                CaseInfo caseInfo, double temperature) {
        super( generator, caseInfo, temperature );
        this.arcCost =  Math.log(arcProb / (1.0 - arcProb));
    }
    
    /**
     * Toggle existence of two arcs into a single node.
     */
    public boolean transform(TOM tom, double ljp) {
        int numNodes = tom.getNumNodes();
        
        // At least 3 nodes are required to do a double skeletal change.
        if (numNodes < 3) {
            return false;
        }
        
        // node[1..3] must be chosen so that they are all different, and node3 follows 
        // node1&2 in the temporal ordering. 
        int node1 = 0, node2 = 0, node3 = 0;
        
        // Create node1, node2, node3 such that all are different
        // and node1 > node2 > node3
        node1 = (int)( numNodes * rand.nextDouble() );
        node2 = (int)( (numNodes-1) * rand.nextDouble() );
        node3 = (int)( (numNodes-2) * rand.nextDouble() );
        
        if ( node2 >= node1 ) { node2 ++; }
        if ( node1 < node2 ) { int temp = node1; node1 = node2; node2 = temp; }
        if ( node3 >= node2 ) { node3++; }
        if ( node3 >= node1 ) { node3++; }
        if ( node1 < node3 ) { int temp = node1; node1 = node3; node3 = temp; }
        
        int childVar = tom.nodeAt( node1 );
        int parentVar1 = tom.nodeAt( node2 );
        int parentVar2 = tom.nodeAt( node3 );
        
        changes[0] = childVar;
        
        // The cost of node1 and node2 cannot be changed (as thier parents remain constant).
        // The modification of the parents of node3 means that is may change.
        // so we must find it's original cost before changing it.
        Node childNode = tom.getNode( childVar );    
        int[] originalParents = childNode.parent;
        double oldChildCost = caseInfo.nodeCache.getMMLCost( childNode );
        
        // Check if the mutation will add too many parents to NodeJ    
        //boolean isArc13 = tom.isArc( parentVar1, childVar );    // is there an arc from 1 -> 3
        //boolean isArc23 = tom.isArc( parentVar2, childVar );    // is there an arc from 2 -> 3
        // addArc is the total number of arcs which will be added/removed by this mutation.
        int addArc = 0;
        if ( tom.isArc( parentVar1, childVar ) ) addArc--; else addArc++;
        if ( tom.isArc( parentVar2, childVar ) ) addArc--; else addArc++;
        if ( childNode.getNumParents() + addArc > tom.maxNumParents ) 
            return false;  // return false if too many parents are present.
        
        // Links P1-->C & P2-->C are toggled.
        nodesToggled[0][0] = childVar;
        nodesToggled[0][1] = childVar;
        nodesToggled[1][0] = parentVar1;
        nodesToggled[1][1] = parentVar2;
        
        // Calculate change in TOM (structure) cost caused by toggle.
        double toggleCost = 
            caseInfo.tomCoster.costToToggleArcs(tom,nodesToggled[0],nodesToggled[1]);

        // toggle node1 -> node3 and node2 -> node3
        doubleMutate( tom, parentVar1, parentVar2, childVar);
        
     // for testing
//        if(tom.isAncestor(0, 1) & tom.isAncestor(0, 2) & tom.isAncestor(0, 5) & tom.isAncestor(1, 2)
//        		& tom.isAncestor(1, 5) & tom.isAncestor(3, 4) & tom.isAncestor(4, 5) & tom.isAncestor(5, 6)
//        		& tom.getNumEdges() == 8)
//        	System.out.println("Sampled!!!!!!!!!!!!!!!");
//        String[] nameOrder = {"H","Drought","TreeCond","PesticideUse","PesticideInRiver","RiverFlow","FishAbundance"};
//        String[][] edge_matrix = tom.edgeMatrix(nameOrder, nameOrder);
//        EditDistance ed = new EditDistance(edge_matrix.length, edge_matrix, 0);
////        ArrayList<String[][]> true_structure = TrueStructures2.getTrigger1TrueStructures();
////        
//        
//        String[][] Native_Fish_latent_6 = new String[7][7];
//		String[] Native_Fish_latent_6_0 = {"null","null","Arrow","null","Arrow","Arrow","null"};
//		String[] Native_Fish_latent_6_1 = {"null","null","Arrow","null","null","Arrow","null"};
//		String[] Native_Fish_latent_6_2 = {"Tail","Tail","null","null","null","null","null"};
//		String[] Native_Fish_latent_6_3 = {"null","null","null","null","Arrow","null","null"};
//		String[] Native_Fish_latent_6_4 = {"Tail","null","null","Tail","null","null","Arrow"};
//		String[] Native_Fish_latent_6_5 = {"Tail","Tail","null","null","null","null","Arrow"};
//		String[] Native_Fish_latent_6_6 = {"null","null","null","null","Tail","Tail","null"};
//		Native_Fish_latent_6[0] = Native_Fish_latent_6_0;
//		Native_Fish_latent_6[1] = Native_Fish_latent_6_1;
//		Native_Fish_latent_6[2] = Native_Fish_latent_6_2;
//		Native_Fish_latent_6[3] = Native_Fish_latent_6_3;
//		Native_Fish_latent_6[4] = Native_Fish_latent_6_4;
//		Native_Fish_latent_6[5] = Native_Fish_latent_6_5;
//		Native_Fish_latent_6[6] = Native_Fish_latent_6_6;
//        
//        ed.calculateEditDistance(Native_Fish_latent_6);
//        int result = ed.getEditDistance1();
//       
//        File log = new File("//ad.monash.edu//home//User005//xzha270//Desktop//ed_result.txt");
//        File log1 = new File("//ad.monash.edu//home//User005//xzha270//Desktop//ed_accepted_result.txt");
//        if(result<=5)
//        {
//        	
//            FileWriter fileWriter = null;
//    		try {
//    			fileWriter = new FileWriter(log, true);
//    		} catch (IOException e) {
//    			// TODO Auto-generated catch block
//    			e.printStackTrace();
//    		}
//    		BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
//    		try {
//    			bufferedWriter.write("Double Skeleta,"+result + "\n");
//    			bufferedWriter.close();
//    		} catch (IOException e) {
//    			// TODO Auto-generated catch block
//    			e.printStackTrace();
//    		}
//    		
////    		System.out.println("Write");
//            
//        }
       
        // update the parents of node3 and find the cost.
        double newChildCost = caseInfo.nodeCache.getMMLCost( childNode );
        
        // The new cost is original - (cost for old arcs and old node) 
        //                          + (cost for new arcs and new node)
        oldCost = 0;
        cost = newChildCost - oldChildCost + toggleCost;
        
        if (accept()) {
            if (caseInfo.updateArcWeights){
                if(tom.isArc(childVar,parentVar1)) {
                    caseInfo.arcWeights[childVar][parentVar1] -= caseInfo.totalWeight;                
                }                
                else {
                    caseInfo.arcWeights[childVar][parentVar1] += caseInfo.totalWeight;
                }

                if(tom.isArc(childVar,parentVar2)) {
                    caseInfo.arcWeights[childVar][parentVar2] -= caseInfo.totalWeight;                
                }                
                else {
                    caseInfo.arcWeights[childVar][parentVar2] += caseInfo.totalWeight;
                }
            }

//            if(result<=5)
//            {
//            	// for testing
//                FileWriter fileWriter1 = null;
//        		try {
//        			fileWriter1 = new FileWriter(log1, true);
//        		} catch (IOException e) {
//        			// TODO Auto-generated catch block
//        			e.printStackTrace();
//        		}
//        		BufferedWriter bufferedWriter1 = new BufferedWriter(fileWriter1);
//        		try {
//        			bufferedWriter1.write("Double Skeleta," + result + "\n");
//        			bufferedWriter1.close();
//        		} catch (IOException e) {
//        			// TODO Auto-generated catch block
//        			e.printStackTrace();
//        		}
                
//        		System.out.println("Write");
//            }
            
            
            
            return true;
        }
        else {
            // toggle node1 -> node3 and node2 -> node3 back to their original state.
            doubleMutate( tom, parentVar1, parentVar2, childVar);
            childNode.parent = originalParents;
            return false;
        }
    }   
}
