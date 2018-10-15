module Assignment3
   
class BinomialModel
  def func dataset, w
    mu = w["mu"]
    dataset.inject(0.0) do |u,row| 
      x = row["label"]
      u -= Math.log((mu ** x) * ((1 - mu) ** (1 - x)))
    end
  end
  
  def grad dataset, w
    mu = w["mu"]
    g = Hash.new {|h,k| h[k] = 0.0}
    dataset.each do |row| 
      x = row["label"]
      g["mu"] -= (x / mu) - (1 - x) / (1 - mu) 
    end

    return g
  end
  
  ## Adjusts the parameter to be within the allowable range
  def adjust w
    w["mu"] = [[0.001, w["mu"]].max, 0.999].min
  end
end
    
class NaiveBayesModel
  def func dataset, w
    -dataset.inject(0.0) do |u, row| 
      cls = row["label"].to_f > 0 ? "pos" : "neg"
      p = cls == "pos" ? 1.0 : 0.0      
      u += Math.log((w["pos_bias"] ** p) * ((1 - w["pos_bias"]) ** (1 - p)))
      n = cls == "neg" ? 1.0 : 0.0      
      u += Math.log((w["neg_bias"] ** n) * ((1 - w["neg_bias"]) ** (1 - n)))
      
      u += row["features"].keys.inject(Math.log(w["#{cls}_bias"])) do |u, fname|
        u += Math.log(w["#{cls}_#{fname}"]) * row["features"][fname]
      end
    end
  end
  
  def grad dataset, w
    g = Hash.new {|h,k| h[k] = 0.0}
    dataset.each do |row|       
      cls = row["label"].to_f > 0 ? "pos" : "neg"
      p = cls == "pos" ? 1.0 : 0.0      
      g["pos_bias"] -= (p / w["pos_bias"]) - (1 - p) / (1 - w["pos_bias"])
      
      n = cls == "neg" ? 1.0 : 0.0      
      g["neg_bias"] -= (n / w["neg_bias"]) - (1 - n) / (1 - w["neg_bias"])

      
      row["features"].each_key do |fname|
        g["#{cls}_#{fname}"] -= row["features"][fname] / w["#{cls}_#{fname}"]
      end
    end

    return g
  end
  
  def predict w, row
    scores = Hash.new
    
    %w(pos neg).each do |cls|
      scores[cls] = row["features"].keys.inject(Math.log(w["#{cls}_bias"])) do |u, fname|
        u += Math.log(w["#{cls}_#{fname}"]) * row["features"][fname]
      end
    end
    cls = scores.keys.max_by {|cls| scores[cls]}
    lbl = cls == "pos" ? "1" : "0"
    {lbl => scores[cls]}
  end
  
  def adjust w
    w.each_key do |fname|
      w[fname] = [[0.001, w[fname]].max, 0.999].min
    end
  end
end

def coin_dataset(n)
  header = %w(x)
  p = 0.7743
  dataset = []
  n.times do
    outcome = rand < p ? 1.0 : 0.0
    dataset << {"features" => {"bias" => 1.0}, "label" => outcome}
  end
  return [header, dataset]
end
    
def plot x, y
  Daru::DataFrame.new({x: x, y: y}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "X"
    plot.y_label "Y"
  end
end
    
end