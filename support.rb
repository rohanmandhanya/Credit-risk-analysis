module Support
    class StochasticGradientDescent
      attr_reader :weights
      attr_reader :objective
      def initialize obj, w_0, lr = 0.01
        @objective = obj
        @weights = w_0
        @n = 1.0
        @lr = lr
      end
      def update x
        dw = @objective.grad(x, @weights)
        learning_rate = @lr / Math.sqrt(@n)

        dw.each_key do |k|
          @weights[k] -= learning_rate * dw[k]
        end

        @objective.adjust @weights
        @n += 1.0
      end
    end
    class LinearRegressionModel
      def func dataset, w
        dataset.inject(0.0) do |u,row| 
          y = row["label"]
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
          u += 0.5 * (y_hat - y) ** 2.0
        end
      end

      def grad dataset, w
        g = Hash.new {|h,k| h[k] = 0.0}
        dataset.each do |row| 
          y = row["label"]
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
          x.each_key do |k|
            g[k] += (y_hat - y) * x[k]
          end
        end

        g.each_key {|k| g[k] /= dataset.size}
        return g
      end

      ## Adjusts the parameter to be within the allowable range
      def adjust w
      end

      def predict row, w
        x = row["features"]    
        y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]} > 0.0 ? 1 : -1
      end
    end
    class LogisticRegressionModel
      def sigmoid x
        1.0 / (1 + Math.exp(-x))
      end

      def func dataset, w
        dataset.inject(0.0) do |u,row| 
          y = row["label"].to_f > 0 ? 1.0 : -1.0
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}

          u += Math.log(1 + Math.exp(-y * y_hat))
        end
      end

      def grad dataset, w
        g = Hash.new {|h,k| h[k] = 0.0}
        dataset.each do |row| 
          y = row["label"].to_f > 0 ? 1.0 : 0.0
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
          syh = sigmoid(y_hat)
          x.each_key do |k|
            g[k] += (syh - y) * x[k]
          end
        end
        g.each_key {|k| g[k] /= dataset.size}
        return g
      end

      def predict row, w
        x = row["features"]    
        y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
        sigmoid(y_hat)
      end

      ## Adjusts the parameter to be within the allowable range
      def adjust w
      end
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