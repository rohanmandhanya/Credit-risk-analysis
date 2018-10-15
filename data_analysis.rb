module DataAnalysis
@@min = 0 
@@max = 10
temp = 3
    
def valid_float(str)
    !!Float(str) rescue false
end
    
def scale_range arr
  minimum = arr.min
  maximum = arr.max
  denominator = maximum - minimum
  counter1 = 0
  loop do
    break if counter1 >= arr.length
    arr[counter1] = ((1.0*arr[counter1]) - minimum)/denominator
    counter1 +=1
  end
  return arr
end

def ecdf arr
#   arr.sort!
  arr = scale_range arr
  x_length = arr.length
  x_axis = []
  y_axis = []
  
  (0.0).step(1.0,1.0/arr.length) do |tread|
    x_axis << tread
    temp_array = []
    
    arr.each_with_index do |element,ind|
      if element <= tread
          temp_array << element
      end
    end
    y_axis << 1.0*temp_array.length/x_length
  end
  return [x_axis, y_axis]
end

def col_data db, col_name
  col_0 = []
  col_1 = []
  db.execute("select distinct #{col_name} from application_train where target = 0 order by #{col_name} limit 10000") do |row|
    next if row[0].nil? or row[0].eql?("")
    col_0 << row[0]
  end
  db.execute("select distinct #{col_name} from application_train where target = 1 order by #{col_name} limit 10000") do |row|
    next if row[0].nil? or row[0].eql?("")
    col_1 << row[0]
  end
  return [col_0,col_1]
end

def add x,y
  y.each do |element|
    x << element
  end
  return x
end
    
def cod_ecdf db, col_name
  
  col0, col1 = col_data db, col_name
  a0, b0 = ecdf col0
  len0 = a0.length
  a1, b1 = ecdf col1
  len1 = a1.length
  m = add a0, a1
  n = add b0, b1
  return m, n, len0, len1
end
    
def p_log_p(x)
    return x * Math.log2(x)
end

# For a given data set this function returns the count for all labels
def getCount split
counts = Hash.new
split.each {
 |x| 
  if counts[x["label"]] == nil 
    counts[x["label"]] = 1.0
  elsif
    counts[x["label"]] += 1
  end
 }
 return counts
end
    
 
def getCountBoost split,d
counts = Hash.new

split.each_with_index {
 |x,i| 
  if counts[x["label"]] == nil 
    counts[x["label"]] = 1.0*d[i]
  elsif
    counts[x["label"]] += d[i]
  end
 }
 return counts
end
 

def entropy p
  total = p.values.inject(0, :+) + 0.0
  return -p.values.reduce(0.0) do |u, pi|
    u += p_log_p(pi/total)
  end
end

def information_gain x, splits # implemented information gain for multi label dataset
  
  rootNodeCount = getCount x
  rootEntropy =  entropy(rootNodeCount)
  totalMembers = rootNodeCount.values.inject(0, :+) + 0.0
  
  totalChildMembers0 = 0.0
  totalChildEntropy = 0
  childcount0 = 0.0
  childEntropy0 = 0.0
  
  splits.each {
 |x| 
    childcount0 = getCount x
    childEntropy0 = entropy(childcount0)
    totalChildMembers0 = childcount0.values.inject(0, :+) + 0.0
    
    totalChildEntropy += ((totalChildMembers0/totalMembers)*childEntropy0)
 }
  return (rootEntropy-totalChildEntropy)
  
end


def split_on_numeric_value x, k, v # splitting based on a threshold
  firstSet = []
  secondSet = []
  splits = []
  x.each {
    |row| 
     if row["features"][k] < v # should it be less than or equals ?
      firstSet << row
    elsif
      secondSet << row
    end
  }
  return [firstSet,secondSet] 
end


def find_best_split_all_thresholds x # try all the values a feature can take to split 
  bestSplit = []
  bestInfoGain = -1.0
  fname = nil
  bestThreshold = 0 
  
  rootNodeCount = getCount x
  rootEntropy =  entropy(rootNodeCount)
  
  if rootEntropy == 0 # no need to split if all are same labels
    return
  end
  
  x[0]["features"].keys.each{
    |f|
    res = x.sort_by do |item|
      item["features"][f]
    end 
    unique_threshold = []
    

    res.each{
      |row|
      unique_threshold << row["features"][f]
    }
    
    unique_threshold = unique_threshold.uniq # only take unique values

  #  puts unique_threshold
    
    (unique_threshold.length-1).times do |i|
      unique_threshold[i] = (unique_threshold[i] +unique_threshold[i+1] )/2.0 # find thresholds btw uniquie sorted members
    end
   
    unique_threshold.each{
      |row|
          infoGain = information_gain res,  split_on_numeric_value(res, f, row)
    if bestInfoGain == -1
      bestInfoGain = infoGain
      bestSplit = split_on_numeric_value(res, f, row)
      fname = f
      bestThreshold = row
    elsif bestInfoGain < infoGain
      bestInfoGain = infoGain
      bestSplit = split_on_numeric_value(res, f, row)
      fname = f
      bestThreshold = row
    end
      }
    
  }
  #puts "best threshold #{bestThreshold}"
  if bestInfoGain < 0
 #   return bestInfoGain*-1
  end
  
  return [bestSplit, fname , bestInfoGain , bestThreshold]
end

def find_best_split_adaboost x,min,max,feature
   # puts @@min
   # puts @@max
    
 #   puts "debug"
    
    bestThreshold = rand(min,max)
    fname = feature
    
    bestInfoGain = information_gain x,split_on_numeric_value(x, feature, bestThreshold)
    bestSplit = split_on_numeric_value(x, feature, bestThreshold)
    
    
    return [bestSplit, fname , bestInfoGain , bestThreshold]
end 

    
    
    
def find_best_split x
  bestSplit = []
  bestInfoGain = -1
  fname = nil
  bestThreshold = 0 

  x[0]["features"].keys.each{
    |f|
    res = x.sort_by do |item|
      item["features"][f]
    end
  infoGain = information_gain res,res.each_slice(res.length/2).to_a
  if bestInfoGain == -1
    bestInfoGain = infoGain
    bestSplit = res.each_slice(res.length/2).to_a
    fname = f
    
    elsif bestInfoGain < infoGain
      bestInfoGain = infoGain
      bestSplit = res.each_slice(res.length/2).to_a
      fname = f
     # puts res.each_slice(res.length/2).to_a[0]
    end
  }
  return [bestSplit, fname , bestInfoGain]
end



#####START - supporting functions to form tree
def largest_hash_key(hash)
 return hash.max_by{|k,v| v}[0]
end
def smallest_hash_value(hash)
 return hash.min_by{|k,v| v}[1]
end
def largest_hash_value(hash)
 return hash.max_by{|k,v| v}[1]
end
def getErrorRatio(set)
  if set.length == 0
    return 0
  end
  distribution = getCount set
  return smallest_hash_value(distribution) / distribution.values.inject(0, :+) 
end
#####END - supporting functions to form tree
 

# function to build a tree based on binay splits
def form_tree dataSet, depth, maxDepth, errorRatio,min,max,feature,d
  
  
  #### START - exit conditions
  if depth > maxDepth
    return
  end
  if dataSet == nil
    return
  end 
    if dataSet.length < 4 ## used a fixed min number of node members. need to make it as function arg
      return
    end
  if (getErrorRatio(dataSet) < errorRatio) && (depth>3)
    return
  end
  #### END - exit conditions


  node = {}
  node["isTerminal"] = 0
  node ["depth"] = depth
    
  #bestSplit, fname,bestInfoGain,threshold = find_best_split_all_thresholds dataSet
    
    threshold = rand(min..max)
    fname = feature
    bestSplit = split_on_numeric_value(dataSet, feature, threshold)
    bestInfoGain = information_gain dataSet,bestSplit

   # puts "debug nill #{bestSplit[0].size} #{bestSplit[1].size}  #{threshold}"

    #puts [threshold,fname,bestInfoGain,bestSplit.size]
    
  if bestInfoGain == nil || fname == nil
    return
  end
   
    
  node["splitfeature"] = fname
  node["threshold"] = threshold
  node["distribution"] = getCountBoost dataSet,d
        #    puts node["distribution"]

  node["distribution"][1] = node["distribution"][1] * 11
          #  puts node["distribution"]

  
  child = []
  if bestSplit != nil
    bestSplit.each{
      |s|
      temp = form_tree s,depth+1,maxDepth,errorRatio,min,max,feature,d
      if(temp != nil) ## incase the next recursive call returns nil
        child << temp
      else 
        terminalNode = {}
        terminalNode["isTerminal"] = 1
        terminalNode["depth"] = depth+1
        terminalNode["distribution"] = getCountBoost s,d
       # puts terminalNode["distribution"]
          terminalNode["distribution"][1] = (terminalNode["distribution"][1] == nil ? 0 : 11*terminalNode["distribution"][1] )
         # puts terminalNode["distribution"]
        #  puts getCount s
        terminalNode ["decision"] = largest_hash_key(terminalNode["distribution"])
        child << terminalNode
      end  
    }
  end   
  node["children"] = child 
  return node
end


def printTree node
  if node == nil
    return 
  end
  depth = node["depth"] 
  temp =" "
  for i in 1..depth*5 do
    temp = temp+" "
  end 
  if node["isTerminal"] == 0
    puts "#{temp} terminal node : False"
    puts "#{temp} split feature : #{node["splitfeature"]}"
    puts "#{temp} threshold : #{node["threshold"]}"
    puts "#{temp} depth : #{node["depth"]}"
    if  (node.has_key?  "valuetoindex")
        puts "#{temp} valuetoindex : #{node["valuetoindex"]}"
    end
    puts "#{temp} distribution : #{node["distribution"]}"
  else 
    puts "#{temp} terminal node : True"
    if  (node.has_key?  "category")
        puts "#{temp} catergory : #{node["category"]}"
    end
    if  (node.has_key?  "value")
        puts "#{temp} value : #{node["value"]}"
    end
    puts "#{temp} decision : #{node["decision"]}"
    puts "#{temp} depth : #{node["depth"]}"
    puts "#{temp} distribution : #{node["distribution"]}"
  end
  puts " "  
  if node["children"] != nil
    node["children"].each{
        |child|
        printTree child
      }
  end 
  return 
end

# gets decision for one example
def getDecision root, sample
  if root["isTerminal"] == 1
    return root["decision"]
  else
    if sample["features"][root["splitfeature"]].to_f < root["threshold"]
      return getDecision root["children"][0],sample
    else
      return getDecision root["children"][1],sample
    end
  end
end

 # returns confusion matrix - can be be 2x2 for binary classification (spambase) or 3x3 in case of iris
def getTestResults root, splits
  
  confusionMatrix = {}

#       		predicted class1	predicted class2
# actual class1    	TP	            	FN
# actual class2  	FP	            	TF
# TP - true positive
# FP - false positive
  
  splits.each{
      |s|

    predictedClass = getDecision root,s.select { |k,v| k == "features" } 
    actualClass = s["label"]
    if confusionMatrix[actualClass] == nil
      confusionMatrix[actualClass] = {}
      confusionMatrix[actualClass][predictedClass] = 1
    else
      if confusionMatrix[actualClass][predictedClass] == nil
        confusionMatrix[actualClass][predictedClass] = 1
      else
         confusionMatrix[actualClass][predictedClass] += 1
      end
    end
  }

  return confusionMatrix
end

def getAccuracy confusion
  tp = 0.0
  total = 0.0
  confusion.keys.each{
    |row|
      confusion[row].keys.each{
        |column|
        if(row == column)
          tp += confusion[row][column]
          total += confusion[row][column]
        else 
          total += confusion[row][column]
        end
        }
    }
  return (tp*100)/total
end


# we will use the first set for testing/cv and 9 other for training 
def kfold dataSet, errorRatios,totalDiv
  kfold = dataSet.each_slice(dataSet.length/totalDiv).to_a
  testSet = dataSet.each_slice(dataSet.length/totalDiv).to_a[0]
  trainSet = dataSet.each_slice(dataSet.length/totalDiv).to_a[1,totalDiv-1]
    accuracy = []
    errorRatios.each{
      |ratio|
        acc =0.0
        i = 1 
        trainSet.each{
          |kset|
          root = form_tree kset,1,10,ratio
       #   puts "debug nil error #{ratio} i : #{i} "
          confusion = getTestResults root, testSet
         # puts root
          temp = getAccuracy confusion
          acc += getAccuracy confusion
          i +=1
        }
   #   puts "seperator"
    accuracy << acc/(totalDiv-1)
   # puts "accuracy is : #{accuracy} with train error ratio #{ratio}" 
  }
  return accuracy
end




#Reuse the code above to load and split
def split_on_categorical_value x, k # multiway split on category
  oneHotEncode = Hash.new
  count = 0 
  x.each {
    |row| 
     if ! (oneHotEncode.has_key?  row["features"][k])
      oneHotEncode[row["features"][k]] = count
      count += 1
    end
  }
   splits = Array.new(count) { Array.new() }
    x.each {
    |row| 
      splits[oneHotEncode[row["features"][k]]] << row
  }
  return splits
end

def find_best_split_categorical x # multiway best catagorical splits
  bestSplit = []
  bestInfoGain = -1
  fname = nil
  
  count = getCount x
  setEntropy = entropy count
  
  if setEntropy == 0
    return 
  end
  
  x[0]["features"].keys.each{
    |f|
      split = split_on_categorical_value(x,f)
     infoGain = information_gain x, split
  if bestInfoGain == -1
    bestInfoGain = infoGain
    bestSplit = split
    fname = f
    elsif bestInfoGain < infoGain
      bestInfoGain = infoGain
      bestSplit = split
      fname = f
    end
  }
  return [bestSplit, fname , bestInfoGain]
end



def form_tree_categorical dataSet, depth, maxDepth, errorRatio,d
  

  ###### Start of Exit conditions ######
  if depth > maxDepth
    return
  end
  if dataSet == nil
    return
  end
  if dataSet.length < 4
    return
  end
  if (getErrorRatio(dataSet) < errorRatio) && (depth>3)
    return
  end
  ###### End of Exit conditions ######


  node = {}
  node["isTerminal"] = 0
  node ["depth"] = depth
  
 # puts "debug1"
  bestSplit, fname,bestInfoGain = find_best_split_categorical dataSet
 # puts "debug2"

  valueToIndexMap = {}
  i = 0
  if bestSplit != nil
  bestSplit.each{
    |s|
    valueToIndexMap[s[0]["features"][fname]] = i
    i +=1
    }
  end
  
  ## if split nodes are pure we need not split further
  if bestInfoGain == nil || fname == nil
    return
  end
  node["valuetoindex"] = valueToIndexMap
  node["splitfeature"] = fname
  node["distribution"] = getCountBoost dataSet,d
      node["distribution"][1] = node["distribution"][1] * 11


  
  child = []
  if bestSplit != nil
    bestSplit.each{
      |s|
      temp = form_tree_categorical s,depth+1,maxDepth,errorRatio,d
      if(temp != nil)
        child << temp
      else 
        terminalNode = {}
        terminalNode["isTerminal"] = 1
        terminalNode["category"] = fname
        terminalNode["value"] = s[0]["features"][fname]
        terminalNode["depth"] = depth+1
        terminalNode["distribution"] = getCountBoost s,d
        terminalNode["distribution"][1] = (terminalNode["distribution"][1] == nil ? 0 : 11*terminalNode["distribution"][1] )

        debug_count = getCount s
        terminalNode ["decision"] = largest_hash_key(terminalNode["distribution"])
        child << terminalNode
      end
    }
  end
  node["children"] = child 
  

  return node
end



def getDecisionCategorical root, sample
  if root["isTerminal"] == 1
    return root["decision"]
  else
    fname = root["splitfeature"]
    sampleValue = sample["features"][fname]
    index = root["valuetoindex"]
    if index[sampleValue] == nil 
      return largest_hash_key root["distribution"]
    end
    return getDecisionCategorical root["children"][index[sampleValue]],sample
  end

end


def getTestResultsCategorical root, splits
  
  confusionMatrix = {}

#       		predicted class1	predicted class2
# actual class1    	TP	            	FN
# actual class2  	FP	            	TF
# TP - true positive
# FP - false positive

  splits.each{
      |s|
   
    predictedClass = getDecisionCategorical root,s.select { |k,v| k == "features" } 
   

    actualClass = s["label"]
    if confusionMatrix[actualClass] == nil
      confusionMatrix[actualClass] = {}
      confusionMatrix[actualClass][predictedClass] = 1
    else
      if confusionMatrix[actualClass][predictedClass] == nil
        confusionMatrix[actualClass][predictedClass] = 1
      else
         confusionMatrix[actualClass][predictedClass] += 1
      end
    end
  }

  return confusionMatrix
end


def kfoldcategorical dataSet, errorRatios,totalDiv
  kfold = dataSet.each_slice(dataSet.length/totalDiv).to_a
  testSet = dataSet.each_slice(dataSet.length/totalDiv).to_a[0]
  trainSet = dataSet.each_slice(dataSet.length/totalDiv).to_a[1,totalDiv-1]
    accuracy = []
    errorRatios.each{
      |ratio|
        acc =0.0
        i = 1 
        trainSet.each{
          |kset|
          root = form_tree_categorical kset,1,10,ratio

          confusion = getTestResultsCategorical root, testSet
            
          temp = getAccuracy confusion
          acc += getAccuracy confusion
          i +=1
        }

    accuracy << acc/(totalDiv-1)
    #puts "accuracy is : #{accuracy} with train error ratio #{ratio}" 
  }
  return accuracy
end



def oneHotEncodeFeatures x
  oneHotEncode = Hash.new
  count = 0
  
  x[0]["features"].keys.each {
    |key|
    count = 1
    i = 0
    oneHotEncode = Hash.new
     x.each {
       |row|
        if ! (oneHotEncode.has_key?  row["features"][key])
          oneHotEncode[row["features"][key]] = count
          x[i]["features"][key] = count
          count = count*2
        else 
           x[i]["features"][key] =  oneHotEncode[row["features"][key]] 
        end
         i += 1
       }
  }
  
  return x
end
    
    def get_info_gain db,name

  columns = db.execute("pragma table_info(#{name})").map{|x| [x["name"],x["type"]]}

  table = []
  frequency_arr = []
  columns.each{
    |feature|

    if((feature[1] == "NUMERIC" or feature[1] == "INTEGER" or feature[1] == "INT" or feature[1] == "NUM")and (!feature[0].include? ":"))
      x=[]
      frequency = 0.0
      max = db.execute("select max(#{feature[0]}) from #{name} where #{feature[0]} is not \"\" ")[0][0]
      min = db.execute("select min(#{feature[0]}) from #{name} ")[0][0]
      mean = db.execute("select avg(#{feature[0]}) from #{name} ")[0][0]
      count_non_null = db.execute("select count(*) from #{name} where #{feature[0]} is not \"\" ")[0][0]
      total_count = db.execute("select count(*) from #{name}")[0][0]
      frequency = count_non_null.to_f / total_count.to_f
        if ( min == nil or max == nil or max == min)
         next
       end
      range = max.to_f - min.to_f
      best_info_gain = nil
      db.execute("select #{feature[0]},target from #{name}").each{
        |row|
          value = {feature[0]=>row[feature[0]].to_f}
          x << { "features"=>value, "label"=>row["TARGET"]}
        }  
     
      if(range!=0)
      (min..max).step(range/10).each{
        |t|  
          infogain = information_gain x,split_on_numeric_value(x, feature[0], t)
          if(best_info_gain==nil or best_info_gain<infogain)
            best_info_gain = infogain
          end
        }
      end
        
      best_info_gain = best_info_gain == nil ? 0 : best_info_gain
      table << [feature[0],best_info_gain,"NUMERIC",frequency,min,max,range]
      frequency_arr << frequency
    elsif (!feature[0].include? ":")
      x=[]
      db.execute("select #{feature[0]},target from #{name}").each{
        |row|
          value = {feature[0]=>row[feature[0]]}
          x << { "features"=>value, "label"=>row["TARGET"]}
      }  
      bestSplit, fname1 , bestInfoGain  = find_best_split_categorical x
      #bestInfoGain = bestInfoGain == nil ? 0 : bestInfoGain
      frequency = 0
      table << [feature[0],bestInfoGain,"TEXT",frequency]
      frequency_arr << frequency
    end
    }
  return table
end

end
