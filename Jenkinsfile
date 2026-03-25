pipeline {
    agent any

    parameters {
        string(name: 'CUTOFF', defaultValue: '10', description: 'Liczba przykładów do odcięcia w zbiorze danych')
    }

    stages {
        stage('Checkout') {
            steps {
                sshagent(['gitea-ssh-key']) {
                    sh 'git clone git@git.wmi.amu.edu.pl:s465090/inz-uczenia-maszynowego.git || echo "Repo already cloned"'
                }
            }
        }

        stage('Generate Dataset') {
            steps {
                sh './inz-uczenia-maszynowego/scripts/download_and_process.sh $CUTOFF'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'inz-uczenia-maszynowego/data/final_dataset.txt', fingerprint: true
            }
        }
    }
}