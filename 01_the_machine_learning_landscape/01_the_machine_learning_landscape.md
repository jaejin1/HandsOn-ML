## What is Machine Learning?

데이터로 학습을 시키고 무엇이든 간에 좋은 결과를 얻어 내는것이다.

## Types of Machine Learning Systems

#### Supervised/Unsupervised Learning

* **supervised learning**

	관측치 하나마다 정답 레이블이 달려있는 데이터셋을 가지고 모델을 학습시킨다. 대표적인 모델은 다중회귀분석, 로지스틱 회귀분석, 인공신경망 등이 있다.
	
* **Unsupervised learning**

	정답 레이블이 달려 있지 않은 데이터를 대상으로도 사용할수 있다. 모델 스스로 학습한다는 특징을 가지고 있다. 
	* **Clustering**

		데이터들의 특성을 고려해 데이터 집단을 정의하고 데이터 집단의 대표할 수 있는 대표점을 찾는것.
		
	* **Dimensionality reduction**

		높은 Dimension은 알고리즘의 성능에 악영향을 미치는 경우가 많다. 우리가 원하지 않는 방향으로 움직일 가능성이커 데이터의 Dimension을 낮춘다.
		
	* **association rule learning**

		데이터간의 연관 법칙을 찾는다.
	
* **Semisupervised learning**

	간단하게 레이블이 달려있는 데이터와 레이블이 달려있지 않은 데이터를 동시에 사용해서 더 좋은 모델을 만들자는것이다. 
	
* **Reinforcement Learning**

	강화학습으로써 어떤 환경안에서 정의된 에이전트가 현재의 상태를 인식해 선택 가능한 행동들 중 보상을 최대화 (올바른 행동을 할때마다 보상을줌) 하는 행동 혹은 행동 순서를 선택하는 방법. 

#### Batch and Online Learning

* **Batch learning**

	* **full-batch**  

		모든 학습데이터를 사용하여 한번에 갱신
		
	* **mini-batch**

		일반적인 방법으로 소량의 학습데이터를 사용하여 갱신
		
* **Online learning**

	시작할때 모든 정보를 가지고 있지 않고 입력을 차례로 받아들이면서 처리하는 알고리즘이다. 일괄학습 시스템과 달리 점진적으로 학습 할 수 있다. 이를 통해 변화하는 데이터와 자율 시스템 모두에 빠르게 적용가능함. 또한 대용량 데이터에 대한 교육이 가능하다. ( 싱싱한 데이터를 모형에 바로 적용시키는것 )

#### Instance-Based Versus model-Based Learning

* **Instance-based learning**

	이웃한 인스턴스를 식별해서 분류한다. 
	대표적으로 KNN이 있다.
	
* **Model-based learning**

	데이터를 학습을 시키면서 최적의 모델을 만들어서 그 모델에 따라 새로운 데이터의 결과를 이끌어 낸다. 즉 새로운 인스턴스에 대해 잘 일반화 될 수 있도록 모델 매개 변수에 대한 최적값을 검색한다.

## Main Challenges of Machine Learning

* Lack of data
* poor data quality
* nonrepresentative data
* uninformative features
* excessively complex models that overfit the data

	모델이 training data에 대해 우수한 결과가 나왔지만 새 인스턴스가 좋지 않은경우 일반적으로 모델이 training data에 지나치게 잘 맞았다고 한다. 이를 overfitting이라고 한다.
	
	
## Data

#### Training Data

데이터를 학습시킬때 사용하는 데이터 training data로 모델을 만들어서 예측하거나 분류함.
	
#### Test Data

모델이 새 인스턴스에 대해서 오류를 추정하는데 사용한다. 

#### Validation Data

보통 training data에서 validation data를 만든다. validation data는 test data의 오류를 예측하고 최고의 모델을 선택하고 하이퍼 파라미터를 튜닝할 수 있다.

방법은 다음과 같이 3가지가 있다.

* **Cross validation**

	training data를 일정한 크기로 나눈다. 예를들어 3/5는 training data 2/5는 validation data로 나눈다.
	
	안 쓰는데이터가 있으므로 성능이 조금 떨어진다.
	
* **LOOCV**

	일정한 크기로 나누지 않고 1개와 그 나머지로 나눈다. 그 후 1개의 데이터를 모든 데이터에 대해서 반복한다. training data에 n개의 데이터가 있으면 1개를 validation data , n-1개를 training data를 사용하고 n 번 반복하면서 1개의 validation data를 모든 데이터가 사용하게 한다. 
	
	장점은 가장 좋은 성능을 보이고 단점은 시간이 너무 오래걸린다.

* **K-ford**

	LOOCV에서 1개의 데이터를 validation data로 했는데 이를 1개가 아닌 K개의 데이터로 나눈 것.
	
	CV와 LOOCV의 중간으로 좋은 성능과 알맞은 시간이 걸려서 많이 사용한다.