/*
 *  [The "BSD license"]
 *  Copyright (c) 2002-2011, Rodney O'Donnell, Lloyd Allison, Kevin Korb
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
// Test Cases for EnumerateDAGs
//

// File: TestEnumerateDAGs.java
// Author: rodo@dgs.monash.edu.au
// Created on 9/02/2005

package camml.test.core.library;

import camml.core.library.Library;
import camml.plugin.rodoCamml.RodoCammlIO;
import cdms.core.Value;
import junit.framework.*;



/** Test functions associated with EnumerateDAGs.java */
public class TestLibrary extends TestCase {
    
    public static Test suite() 
    {
        return new TestSuite(TestLibrary.class);
    }
    
    
    /** Test Library.weightedSummaryVec */
    public final void testSummaryVec() throws Exception {
        Value.Vector data = RodoCammlIO.load("camml/test/AsiaCases.1000.cas");
        Value.Vector summary = Library.makeWeightedSummaryVec(data);
        
        assertEquals( 37, summary.length() );
    }
    

    /** Test Library.weightedSummaryVec */
    public final void testJoinVec() throws Exception {
        Value.Vector data = RodoCammlIO.load("camml/test/AsiaCases.1000.cas");
        Value.Vector join = Library.joinVectors(data, data.cmpnt(7), "extra" );
            
        assertEquals( join.cmpnt(7), join.cmpnt(8) );
        System.out.println("join.t = " + join.t);
    }

}
