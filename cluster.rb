
require 'json'
require 'daru'
require 'distribution'

n = 100
clusters = [[-4,4], [-4,-4], [4,4], [4,-4], [0,0]]
data = []

clusters.each.with_index do |mean, i|  
  r = mean.collect {|m| Distribution::Normal.rng(m,1)}
  n.times do |j|
    data << {"features" => {"x1" => r[0].call(), "x2" => r[1].call()}, "label" => i, "cluster" => i}
  end
end

def plot_clusters(data)
  x1 = []
  x2 = []
  target = []
  data.each do |row|
    x1 << row["features"]["x1"]
    x2 << row["features"]["x2"]
    target << row["cluster"]
  end
  df = Daru::DataFrame.new({x1: x1, x2: x2, target: target})
  df.to_category :target
  df.plot(type: :scatter, x: :x1, y: :x2, categorized: {by: :target, method: :color}) do |plot, diagram|
    plot.xrange [-8,8]
    plot.x_label "X1"
    plot.yrange [-8,8]  
    plot.y_label "X2"
    plot.legend false
  end
end

plot_clusters(data)

def init_cluster data, k
  means = Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0.0}}
    max = Hash.new {|h,k| h[k] = 0.0}
    min = Hash.new {|h,k| h[k] = 0.0}
    data.each do |row|
      row["features"].each do |key,val|
        if max[key] < val
          max[key] = val
        end
        if min[key] > val
          min[key] = val
        end
      end
    end
  
  k.times do |cnt|
    keys = max.keys
    temp_hash = Hash.new
    keys.each do |key|
      ma = max[key]
      mi = min[key]
      range = ma-mi
      temp_hash[key] = rand(range) + mi
    end
    means[cnt] = temp_hash
  end
  
  means
end

means = init_cluster data, 5

def assign_cluster(data, means)
  cluster_array = []
  data.each{|row|
    close_dist = Array.new(means.length, 0.0)
    row["features"].each{|x|
      means.each.with_index{|m|
        close_dist[m[0]] += (m[1][x[0]] - x[1])**2.0
        }
      }
    close_min_dist = close_dist.min
    row["cluster"] = close_dist.each_with_index.min[1]
      cluster_array << close_dist.collect{|d|
      d > close_min_dist ? 0.0 : 1.0}
    }
  return cluster_array
end
d = data.sample(50)
z = assign_cluster(d, means)

plot_clusters(d)

def calculate_means z, data
 k = z.first.length
  mean = Hash.new {|h,k| h[k] = Hash.new {|h,k| h[k] = 0.0}}
  sum = [0.0]*k
  data.each_with_index do |row, i|
    zoi = z[i]
    for j in (0..k-1) do
      sum[j] += zoi[j]
      row["features"].each_key do |f|
        mean[j][f] += row["features"][f] * zoi[j]
      end
    end
  end
  for j in (0..k-1) do
    mean[j].each_key {|key| mean[j][key] /= sum[j] unless sum[j].zero?}
  end
  return mean
end

calculate_means(z, d)

def cluster_dist(m0, m1)
  total = 0.0
  m0.keys.each{|key0|
    m0.first[1].keys.each{|key1|
      total += (m0[key0][key1] - m1[key0][key1])**2
      }
    }
  return total / m0.length
end

def k_means(data, k)

  all_dist = []
  m0 = init_cluster data, k
  z = assign_cluster data, m0
  
  for n in (0..100) do
    
    m1 = calculate_means z, data
    dist = cluster_dist m0, m1
      break if dist <= 0.001
        m0 = m1
        z = assign_cluster data, m0
        all_dist << dist
  
  end
    
  return [all_dist, m0]
end

dists, means, z = k_means data, 3
iters = Array.new(dists.size) {|i| i }

df = Daru::DataFrame.new({iters: iters, dists: dists})
puts dists
df.plot(type: :line, x: :iters, y: :dists) do |plot, diagram|
  plot.x_label "X"
  plot.y_label "Mean Dist"
  diagram.title "Cluster Convergence"
  plot.legend false
end
