

class EKF_Slam
{
    public:
        EKF_Slam();
        ~EKF_Slam();

        void initEKF();
        void PredictionStep();
        void CorrectionStep();
};