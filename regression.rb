module Regression

def labelData dataset, column_type
  data = []
  
  feature_values = Hash.new { |h,k| h[k] = []}
  
  dataset.each do |row|
    temp_hash = Hash.new
    features = Hash.new
    row.each do |k,v|
      if column_type[k].eql?("TEXT")
        if k.eql?("TARGET") || k.eql?("target")
          temp_hash["label"] = v
            next
        end
        
        if !(feature_values[k].index(v).nil?)
          features[k] = feature_values[k].index(v)*1.0
        else
          feature_values[k] << v
          features[k] = feature_values[k].index(v).*1.0
        end
      else
#       elsif column_type[k].eql?("INTEGER")
        if k.eql?("TARGET") || k.eql?("target")
          temp_hash["label"] = v
            next
        end
        if v.nil? or v ==""
#             features[k] = nil
          else
          features[k] = v*1.0
          end
      end
    end
    temp_hash["features"] = features
    data << temp_hash
  end
  return data
end

def update_weights(w, dw, learning_rate)
  w1 = w.clone
  dw.keys.each {|key|
     w1[key] -= learning_rate * dw[key]
    }
  return w1
end

def ext_column dataset
  corrected_data = []
  dataset.each do |row|
    temp_row  = Hash.new
    features = Hash.new {|h,k| h[k]=0}
    row.each do |j,k|
      if j.class == String
        if j.eql?("TARGET") || j.eql?("target")
          temp_row["label"] = k
        else
          if j.include?("EXT_SOURCE")
            if row[j] == nil or row[j] ==""
              features[j] = 0.0
            else
              features[j] = k.ceil(1)
            end
          else
            features[j]  = k
          end
        end
        temp_row["features"] = features
      end
    end
    corrected_data << temp_row
  end
  return corrected_data
end
  
def score_model model, db, w, sql
  scores = []
  db.execute(sql) do |row|
    corrected_row = ext_column([row])[0]
    prediction = model.predict(corrected_row, w)
    scores << [prediction,corrected_row["label"]]
  end
  return scores
end
    
    
def roc_curve(scores)
      total_neg = scores.inject(0.0) {|u,s| u += (1 - s.last)}
      total_pos = scores.inject(0.0) {|u,s| u += s.last}
      c_neg = 0.0
      c_pos = 0.0
      fp = [0.0]
      tp = [0.0]
      auc = 0.0
      scores.sort_by {|s| -s.first}.each do |s|
        c_neg += 1 if s.last <= 0
        c_pos += 1 if s.last > 0  

        fpr = c_neg / total_neg
        tpr = c_pos / total_pos
        auc += 0.5 * (tpr + tp.last) * (fpr - fp.last)
        fp << fpr
        tp << tpr
   end
   return [fp, tp, auc]
end
end


