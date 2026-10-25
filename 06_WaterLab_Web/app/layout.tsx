import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
 title: "WaterLab · 智能需水预测工作台",
 description: "上传数据、运行PSO-BP预测，探索未来情景并与AI对话。",
 icons: {icon:"/favicon.svg"}
};
export default function RootLayout({children}:{children:React.ReactNode}) {
 return <html lang="zh-CN"><body>{children}</body></html>;
}
