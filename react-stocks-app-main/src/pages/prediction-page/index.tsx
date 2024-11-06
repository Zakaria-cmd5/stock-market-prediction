import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useForm } from "react-hook-form";
import axios from "axios";
import PredictChart from "@/components/chart/predict";
import useDocumentTitle from "@/hooks/useDocumentTitle";

interface FormShape {
  comp: string;
  days: number;
}

const PredictPage = () => {
  useDocumentTitle("Predict")

  const { register, handleSubmit } = useForm<FormShape>();
  const [predictionData, setPredictionData] = useState<
    { day: number; value: number }[]
  >([]);
  const [imgResponse, setImgResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const onSubmit = async (data: FormShape) => {
    try {
      setError(false)
      setLoading(true);
      const response = await axios.post("http://127.0.0.1:5000/predict", data);
      const formattedData = response.data.prediction.map(
        (value: number, index: number) => ({
          day: index + 1,
          value,
        })
      );
      setPredictionData(formattedData);
      setImgResponse(response.data.image);
      setLoading(false);
    } catch (error: any) {
      setError(error);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 p-5 md:p-10">
      <form
        className="space-y-3 max-w-xl pl-10"
        onSubmit={handleSubmit(onSubmit)}
      >
        <Input
          placeholder="Please choose what company you want to predict"
          {...register("comp")}
          className="w-full"
        />
        <Input
          placeholder="Please choose how many days you want to predict"
          type="number"
          {...register("days", { required: true })}
          className="w-full"
        />
        <Button disabled={loading} className="w-auto">
          {loading ? "Predicting..." : "Predict"}
        </Button>
      </form>
      {error && <p className="font-semibold ml-10 error-color">An error has been occuerd</p>}
      {predictionData.length > 0 && (
        <div className="w-full">
          <img
            src={`data:image/jpeg;base64,${imgResponse}`}
            className="p-10"
            alt="Prediction Graph"
          />
        </div>
      )}
      {predictionData.length > 0 && (
        <div className="w-full">
          <PredictChart data={predictionData} className="w-full max-w-4xl" />
        </div>
      )}
    </div>
  );
};

export default PredictPage;
