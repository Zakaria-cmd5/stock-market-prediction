import React from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface Props {
  data: { day: number; value: number }[];
  className: string;
}

const PredictChart: React.FC<Props> = ({ data, className }: Props) => {
  return (
    <div className={`max-w-6xl ${className}`}>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <XAxis
            dataKey="day"
            label={{ value: "Days", position: "insideBottomRight", offset: 0 }}
          />
          <YAxis
            label={{ value: "Price", angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <CartesianGrid strokeDasharray="3 3" />
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictChart;
