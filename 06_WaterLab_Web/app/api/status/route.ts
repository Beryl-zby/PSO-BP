import {getModelConfig} from '../../../lib/qwen';
export async function GET(){const config=getModelConfig();return Response.json({connected:config.configured,model:config.model||'尚未配置模型',local:config.local},{headers:{'Cache-Control':'no-store'}});}
