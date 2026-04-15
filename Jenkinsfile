pipeline {
    agent any

    parameters {
        string(name: 'EPOCHS', defaultValue: '30', description: 'Liczba epok')
    }

    stages {

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("diabetes-model")
                }
            }
        }

        stage('Train') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.image("diabetes-model").inside {
                        sh """
                        echo "TRAINING | EPOCHS=${EPOCHS}"
                        python train.py --epochs ${EPOCHS}
                        """
                    }
                }
            }
        }

        stage('Eval') {
            when {
                branch 'eval'
            }
            steps {
                script {

                    copyArtifacts(
                        projectName: env.JOB_NAME,
                        selector: lastSuccessful()
                    )

                    docker.image("diabetes-model").inside {
                        sh """
                        echo "EVALUATION"
                        python predict.py
                        """
                    }
                }
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: '*.csv, *.pkl, *.pth', fingerprint: true
        }
    }
}