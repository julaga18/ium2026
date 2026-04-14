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

        stage('Run') {
            steps {
                script {
                    docker.image("diabetes-model").inside {

                        if (env.BRANCH_NAME == 'main') {
                            sh """
                            echo "TRAINING | EPOCHS=${EPOCHS}"
                            uv run python train.py --epochs ${EPOCHS}
                            """
                        }

                        else if (env.BRANCH_NAME == 'eval') {
                            sh """
                            echo "EVALUATION"
                            uv run python predict.py
                            """
                        }
                    }
                }
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: '*.pth, *.csv, *.txt', fingerprint: true
        }
    }
}