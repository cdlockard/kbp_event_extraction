package edu.washington.cs.knowitall.kbp2014.multir.slotfiller


import java.io._
import java.nio.file.{Paths, Files}

import scala.collection.JavaConversions._
import scala.io.Source

import com.typesafe.config.ConfigFactory

/* 
 * For UWashington1 run, we combine the results from OpenIE, ImplIE, and Multir-PERLOC
 * The confidence scores are assumed normalized (between 0 and 1),
 * so that this output can be submitted directly.
 */
object CombineKBPResults {

  val config = ConfigFactory.load("kbp-2015-combine-results.conf")
  val queriesFileName = config.getString("queries-file")
  val openieFileName = config.getString("openie-file")
  val implieFileName = config.getString("implie-file")
  val multirFileName = config.getString("multir-file")
  val outFileName = config.getString("combined-results-file")
  val outStatsFileName = config.getString("out-stats-file")
  val runID = config.getString("runID")
  val roundID = config.getString("roundID")
  
  case class kbpAnswer(id: String, rel: String, runID: String, provAll: String, 
      slotFill: String, slotFillType: String, provSF: String, confScore: String)
  
      
  def main(args: Array[String]) {
  
    val outStream = new PrintStream(outFileName)
    val outStatsStream = new PrintStream(outStatsFileName)
    
    // ---------------------------
    // Read Openie Slot Fills
    // ---------------------------
    val openieSF = {
     
      val inputFilename = openieFileName    
      // Does file exist?
      if (!Files.exists(Paths.get(inputFilename))) {
        System.out.println(s"Openie file $inputFilename doesn't exist!  " + s"Exiting...")
        sys.exit(1)
      } 
      // Read file, line by line
      Source.fromFile(inputFilename).getLines().map(line => {
        val tokens = line.trim.split("\t")
        if(tokens.size >= 8){ 
          kbpAnswer(tokens(0), tokens(1), tokens(2), tokens(3), tokens(4), tokens(5), 
            tokens(6), tokens(7))
        }
        else{
          kbpAnswer("token", "tokens(1)", "tokens(2)", "tokens(3)", "tokens(4)", "tokens(5)", 
            "tokens(6)", "tokens(7)")
        }
      }).toList.filter(k => k.id != "token")
    }           

    // ---------------------------
    // Read Implie Slot Fills
    // ---------------------------
    val implieSF = {
     
      val inputFilename = implieFileName    
      // Does file exist?
      if (!Files.exists(Paths.get(inputFilename))) {
        System.out.println(s"Implie file $inputFilename doesn't exist!  " + s"Exiting...")
        sys.exit(1)
      } 
      // Read file, line by line
      Source.fromFile(inputFilename).getLines().map(line => {
        val tokens = line.trim.split("\t")
        if(tokens.size >= 8){ 
          kbpAnswer(tokens(0), tokens(1), tokens(2), tokens(3), tokens(4), tokens(5), 
            tokens(6), tokens(7))
        }
        else{
          kbpAnswer("token", "tokens(1)", "tokens(2)", "tokens(3)", "tokens(4)", "tokens(5)", 
            "tokens(6)", "tokens(7)")
        }
      }).toList.filter(k => k.id != "token")      
    }           

    // ---------------------------
    // Read Multir Slot Fills
    // ---------------------------
    val multirSF = {
     
      val inputFilename = multirFileName    
      // Does file exist?
      if (!Files.exists(Paths.get(inputFilename))) {
        System.out.println(s"Multir file $inputFilename doesn't exist!  " + s"Exiting...")
        sys.exit(1)
      } 
      // Read file, line by line
      Source.fromFile(inputFilename).getLines().map(line => {
        val tokens = line.trim.split("\t")
        if(tokens.size >= 8){ 
          kbpAnswer(tokens(0), tokens(1), tokens(2), tokens(3), tokens(4), tokens(5), 
            tokens(6), tokens(7))
        }
        else{
          kbpAnswer("token", "tokens(1)", "tokens(2)", "tokens(3)", "tokens(4)", "tokens(5)", 
            "tokens(6)", "tokens(7)")
        }
      }).toList.filter(k => k.id != "token")
    }           
    
    println("openieSF size: " + openieSF.size)
    println("implieSF size: " + implieSF.size)
    println("multirSF size: " + multirSF.size)    
    
    // ---------------------------
    // Parse the Queries
    // ---------------------------
    
    val queries = KBPQuery.parseKBPQueries(queriesFileName,roundID)
    println("Number of queries: " + queries.size)
    
    queries.foreach(q => {
      
      //There is one slot to fill
      if(q.slotsToFill.size == 1){
        
        val slotToFill = q.slotsToFill.toList(0)        
        val numAnswers = slotToFill.maxResults         
        
        val openieList = openieSF.filter(s => s.id == q.id && s.rel == slotToFill.name).sortBy(s => s.confScore)         
        val implieList = implieSF.filter(s => s.id == q.id && s.rel == slotToFill.name).sortBy(s => s.confScore)         
        val multirList = multirSF.filter(s => s.id == q.id && s.rel == slotToFill.name).sortBy(s => s.confScore)         
        
        outStatsStream.print(q.id + " " + q.name)
        outStatsStream.print(" numAnswers: " + numAnswers + " " + slotToFill.name)
        outStatsStream.println(" o i m size: " + openieList.size + " " + implieList.size + " " + multirList.size) 
        
        var slotSet: Set[kbpAnswer] = Set()
        var slotFillSet: Set[String] = Set()
        
        val openieSlotSet = openieList.size match {
          case 0 => Set()
          // if want to limit the answers used
          //case s if s < numAnswers => openieList.toSet.filter(s => !slotFillSet.contains(s.slotFill))
          //case s if s >= numAnswers => openieList.dropRight(openieList.size-numAnswers).
          //  toSet.filter(s => !slotFillSet.contains(s.slotFill))         
          case s => openieList.toSet.filter(s => !slotFillSet.contains(s.slotFill.trim.toLowerCase))  
        }
        openieSlotSet.foreach(s => slotFillSet += s.slotFill.trim.toLowerCase)                
 
        val implieSlotSet = implieList.size match {
          case 0 => Set()
          // if want to limit the answers used
          //case s if s < numAnswers => implieList.toSet.filter(s => !slotFillSet.contains(s.slotFill))
          //case s if s >= numAnswers => implieList.dropRight(implieList.size-numAnswers).
          //  toSet.filter(s => !slotFillSet.contains(s.slotFill))         
          case s => implieList.toSet.filter(s => !slotFillSet.contains(s.slotFill.trim.toLowerCase))
        }
        implieSlotSet.foreach(s => slotFillSet += s.slotFill.trim.toLowerCase)     

        val multirSlotSet = multirList.size match {
          case 0 => Set()
          // if want to limit the answers used
          //case s if s < numAnswers => multirList.toSet.filter(s => !slotFillSet.contains(s.slotFill))
          //case s if s >= numAnswers => multirList.dropRight(multirList.size-numAnswers).
          //  toSet.filter(s => !slotFillSet.contains(s.slotFill))         
          case s => multirList.toSet.filter(s => !slotFillSet.contains(s.slotFill.trim.toLowerCase))
        }
        multirSlotSet.foreach(s => slotFillSet += s.slotFill.trim.toLowerCase)     
        
        // -----------------------------------------------------------------------------------------
        // Combine the answers from the 3 systems
        //
        // For single-value slots, such as org:city_of_HQ, take answer from openie (if exists), 
        // else from implie (if exists), else from multir (if exists)
        //
        // For list-value slots, take all answers from all 3 systems (for recall) with 
        // with deduplication; i.e. take answers from openie first, then add implie slotfills that
        // aren't already in the set, then add multir slotfills that aren't already in the set
        // -----------------------------------------------------------------------------------------
        if(numAnswers==1){
          if(openieList.size > 0) slotSet = Set(openieList(0))
          else if(implieList.size > 0) slotSet = Set(implieList(0))         
          else if(multirList.size > 0) slotSet = Set(multirList(0))
        }
        else{       
          slotSet = openieSlotSet ++ implieSlotSet ++ multirSlotSet
        }
        
        // -----------------------------
        // Write out combined answers
        // -----------------------------
                
        slotSet.size match {
          // answer is NIL, don't need to print out for Cold Start  
          case 0 => //outStream.println(q.id + "\t" + slotToFill.name + "\t" + runID + "\t" + "NIL")
          // non-NIL answers
          case _ => slotSet.foreach(s => outStream.println(s.id + "\t" + s.rel + "\t" + runID + "\t" + 
            s.provAll + "\t" + s.slotFill + "\t" + s.slotFillType + "\t" + s.provSF + "\t" + s.confScore))
        }                           
        
      }
      //if Something didn't parse correctly, and we don't have exactly one slot identified to fill
      else{
        outStatsStream.println("ERROR: " + " " + q.name + " " + "Slots to fill: " + q.slotsToFill.size)
      }
      
    })
    
    outStream.close()
    outStatsStream.close()
    println("closed output streams")
    
  }  
}
  
